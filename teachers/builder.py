import os
import logging
from collections import OrderedDict
from typing import List, Dict, Union

import torch

from .teachers_config import TEACHER_CFG
from dinov2.models.aggregation import DinoV2LargeThreeFAM


logger = logging.getLogger()


def build_teachers(
    teacher_names: List[str],
) -> Union[Dict[str, torch.nn.Module], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    teachers = OrderedDict()
    teacher_ft_stats = OrderedDict()

    for tname in teacher_names:
        tname = tname.strip()
        logger.info("Loading teacher '{}'".format(tname))
        model = _build_teacher(tname)
        teachers[tname] = model

        # buffers for teacher feature statistics
        ft_dim = TEACHER_CFG[tname]["num_features"]

        if isinstance(model, DinoV2LargeThreeFAM):
            logger.info(f"Teacher '{tname}' is multi-head (A/B/C): creating separate stats for each head")
            for suffix in ["A", "B", "C"]:
                teacher_ft_stats[f"{tname}_{suffix}"] = {
                    "cls": {
                        "mean": torch.zeros(1, ft_dim).cuda(),
                        "std": torch.ones(1, ft_dim).cuda(),
                    },
                    "patch": {
                        "mean": torch.zeros(1, 1, ft_dim).cuda(),
                        "std": torch.ones(1, 1, ft_dim).cuda(),
                    },
                }
        else:
            teacher_ft_stats[tname] = {
                "cls": {
                    "mean": torch.zeros(1, ft_dim).cuda(),
                    "std": torch.ones(1, ft_dim).cuda(),
                },
                "patch": {
                    "mean": torch.zeros(1, 1, ft_dim).cuda(),
                    "std": torch.ones(1, 1, ft_dim).cuda(),
                },
            }

    teacher_dims = [TEACHER_CFG[t]["num_features"] for t in teacher_names]

    return teachers, teacher_ft_stats, teacher_dims



def _build_teacher(name):
    # name is expected to be in the following format:
    #  dino_vitbase_16
    #  <model_name>_<arch>_<patch_size>
    if name not in TEACHER_CFG.keys():
        raise ValueError(
            "Unsupported teacher name: {} (supported ones: {})".format(
                name, TEACHER_CFG.keys()
            )
        )

    cfg        = TEACHER_CFG[name]
    ckpt_path  = cfg.get("ckpt_path", "")
    ckpt_key   = cfg.get("ckpt_key", None)
    loader_fn  = cfg["loader"]    

    
    model = loader_fn(ckpt_path)

    if ckpt_path and os.path.isfile(ckpt_path):
        logger.info(f"Loading checkpoint for teacher '{name}' from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd   = ckpt[ckpt_key] if ckpt_key in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    else:
        if ckpt_path:
            logger.warning(
                f"Checkpoint for teacher '{name}' not found at '{ckpt_path}'. "
                "The model will start with random weights."
            )

        for param in model.parameters():
            param.requires_grad = False

    model = model.cuda()
    logger.info(model)

    return model
