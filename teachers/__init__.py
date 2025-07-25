import logging
from collections import defaultdict
from typing import List, Dict

import torch

from utils import standard_normalize
from .teachers_config import TEACHER_CFG
from teachers.config import CONFIG
from .builder import build_teachers
from .concat import RepresentationAlignmentBlock, AttentionFusionBlock


logger = logging.getLogger()


def get_teacher_output(
    image: torch.Tensor,
    teachers: Dict[str, torch.nn.Module],
    teacher_ft_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    teacher_ft_stat_ema_momentum: float = 0.0,
    strategy: List[str] = None,
    aggregation_parameter: Dict[str, float] = None,
    aggregator=None,
    use_fp16=True
) -> Dict[str, Dict[str, torch.Tensor]]:

    teacher_output = defaultdict(dict)
    cls_list, patch_list = [], []

    use_mean = strategy and "mean" in strategy
    use_rab = strategy and "rab" in strategy
    use_abf = strategy and "abf" in strategy

  

    # ---------------------------------------------
    # 1.  Forward dei singoli teacher   (no-grad)
    # ---------------------------------------------
    with torch.no_grad():
        for tname in teachers.keys():
            amp_enable = bool(use_fp16) and "vit_tiny" not in tname.lower()
            with torch.cuda.amp.autocast(enabled=amp_enable):
                image_copy = image

                finetuning_bands = TEACHER_CFG[tname]["finetuning_bands"]
       
                device = "cuda"
                mean = CONFIG[finetuning_bands]["mean"].to(device)
                std = CONFIG[finetuning_bands]["std"].to(device)
                bands = CONFIG[finetuning_bands]["bands"]
          
                
                image_copy = image_copy[:, bands, :, :]
                image_copy = (image_copy - mean) / std
                tout_dict = teachers[tname].forward_features(image_copy)

                #da qui nuovo
                if set(tout_dict.keys()) == {"A", "B", "C"}:
                    for tt in {"A", "B", "C"}:
                        for ttype in ["cls", "patch"]:
                            tout = tout_dict[tt][ttype]

                            mean_ema = teacher_ft_stats[f"{tname}_{tt}"][ttype]["mean"]
                            std_ema = teacher_ft_stats[f"{tname}_{tt}"][ttype]["std"]
                          
                        
                            if ttype == "cls":
                                if tout.ndim == 3 and tout.shape[1] == 1:
                                    tout = tout[:, 0, :]  # [B, 1, D] ? [B, D]
                                elif tout.ndim == 1:
                                    tout = tout.unsqueeze(0)  # [D] ? [1, D]
                                    
                                assert tout.shape[0] == image.shape[0], f"Expected batch {image.shape[0]}, got {tout.shape[0]}"


                            if ttype == "patch":
                                if tout.ndim == 2:
                                    tout = tout.unsqueeze(0)  # [N, D] ? [1, N, D]
                                
                                # ?? sicurezza: dev'essere [B, N, D]
                                assert tout.ndim == 3, f"Expected [B, N, D], got {tout.shape}"
                                assert tout.shape[0] == image.shape[0], f"Expected batch {image.shape[0]}, got {tout.shape[0]}"

                            
                            
                            tout = standard_normalize(
                                tout,
                                mean_ema=mean_ema,
                                std_ema=std_ema,
                                ema_momentum=teacher_ft_stat_ema_momentum,
                            )
                            '''
                            if tout.ndim == 3 and strategy is None:
                                tout = tout.squeeze(1)
                            if tout.ndim == 2 and strategy is not None:
                                tout = tout.unsqueeze(1)'''
                                
                            
                            if use_mean or use_abf:
                                if ttype == "cls":
                                    cls_list.append(tout)
                                else:
                                    patch_list.append(tout)
                            teacher_output[f'{tname}_{tt}'][ttype] = tout
                #a qui nuovo
                else:
                    
                    for ttype in ["cls", "patch"]:
                        key = f"x_norm_{ttype}{'token' if ttype == 'cls' else 'tokens'}"
                        tout = tout_dict[key]  # (B, L, C)
                        tout = standard_normalize(
                            tout,
                            mean_ema=teacher_ft_stats[tname][ttype]["mean"],
                            std_ema=teacher_ft_stats[tname][ttype]["std"],
                            ema_momentum=teacher_ft_stat_ema_momentum,
                        )
                        if tout.ndim == 3 and tout.shape[1] == 1 and strategy is None:
                            tout = tout.squeeze(1)
                            
                        if tout.ndim == 2 and strategy != None:
                            tout = tout.unsqueeze(1)
                            
                        if use_mean or use_abf:
                            if ttype == "cls":
                                cls_list.append(tout)
                            else:
                                patch_list.append(tout)
            
                        teacher_output[tname][ttype] = tout
                    
    # ---------------------------------------------
    # 2.  Fusione con l�aggregator (grad-on)
    # ---------------------------------------------
    merged_output = {}

    if use_mean:
        merged_output["mean"] = {
            "cls": torch.mean(torch.stack(cls_list, dim=0), dim=0).squeeze(1),
            "patch": torch.mean(torch.stack(patch_list, dim=0), dim=0)
        }

    if use_abf:
        assert aggregator is not None, "Pass `aggregator` if using ABF strategy"
        abf = aggregator(cls_list, patch_list)
        merged_output["abf"] = {
            "cls": abf["cls"].squeeze(1),
            "patch": abf["patch"]
        }

    if merged_output:
        teacher_output = {"mergedFeatures": {}}
        # Se entrambe presenti, si pu? decidere se restituirle separate o fonderle ulteriormente
        if "mean" in merged_output and "abf" in merged_output:
            alpha = aggregation_parameter.get("alpha", 0.5)
            beta = aggregation_parameter.get("beta", 0.5)
            assert abs(alpha + beta - 1.0) < 1e-5, f"alpha + beta must be 1.0, got {alpha + beta}"

            teacher_output = {"mergedFeatures": {
                "cls": alpha * merged_output["mean"]["cls"] + beta * merged_output["abf"]["cls"],
                "patch": alpha * merged_output["mean"]["patch"] + beta * merged_output["abf"]["patch"]
            }}
        elif "mean" in merged_output:
            teacher_output = {"mergedFeatures": merged_output["mean"]}
        elif "abf" in merged_output:
            teacher_output = {"mergedFeatures": merged_output["abf"]}
            

    return teacher_output
