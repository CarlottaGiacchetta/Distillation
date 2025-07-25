import math
import os
from collections import defaultdict
from typing import Dict, List, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.models import vision_transformer, grouping
from timesformer import timesformer 
from teachers.config import CONFIG
from dinov2.logging import setup_logging, ExternalLogger, MetricLogger
import logging
from einops import rearrange, reduce, repeat
logger = logging.getLogger()


IMAGENET_URLS = {
    "vit_small": "models/backbonePretrained/dinov2_vits14_reg4_pretrain.pth",
    "vit_large": "models/backbonePretrained/dinov2_vitl14_reg4_pretrain.pth",
}


class UNIC(nn.Module):
    def __init__(self, encoder, lp, in_chans, strategy = 'split', num_frames = 3):
        super().__init__()
        self.encoder = encoder
        self.lp = lp
        self.in_chans = in_chans
        self.strategy = strategy
        self.num_frames = num_frames

    def forward(self, image):
        _, _, H, W = image.shape
        if (H, W) != (224, 224):
            #print('faccio resize delle immagini perchè ho dimensioni: ', (H, W))
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False) # x: (B, 3, 120, 120) → (B, 3, 224, 224)

        if self.encoder.__class__.__name__.startswith("TimeSformer"): 
            if self.num_frames == 3:
                band = "nove"
            elif self.num_frames == 4:
                band = "all"
        elif self.encoder.__class__.__name__.startswith("GroupChannelsVisionTransformer"):
            band = "all"
        else:
            band = "nove"
            
        
        image = image[:, CONFIG[band]['bands'], :]
        std = CONFIG[band]['std']
        mean = CONFIG[band]['mean']
        
        image = (image - mean.to(image.device)) / std.to(image.device)
        
        if self.encoder.__class__.__name__.startswith("TimeSformer"): 
            B, C_all, H, W = image.shape
            image = image.view(B, self.in_chans, self.num_frames, H, W)

            x, T, W = self.encoder.model.patch_embed(image)
            cls_tokens = self.encoder.model.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
    
            ## resizing the positional embeddings in case they don't match the input at inference
            if x.size(1) != self.encoder.model.pos_embed.size(1):
                pos_embed = self.encoder.model.pos_embed
                cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
                other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
                P = int(other_pos_embed.size(2) ** 0.5)
                H = x.size(1) // W
                other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
                new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
                new_pos_embed = new_pos_embed.flatten(2)
                new_pos_embed = new_pos_embed.transpose(1, 2)
                new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
                x = x + new_pos_embed
            else:
                x = x + self.encoder.model.pos_embed
            x = self.encoder.model.pos_drop(x)
    
    
            ## Time Embeddings
            if self.encoder.model.attention_type != 'space_only':
                cls_tokens = x[:B, 0, :].unsqueeze(1)
                x = x[:,1:]
                x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
                ## Resizing time embeddings in case they don't match
                if T != self.encoder.model.time_embed.size(1):
                    time_embed = self.encoder.model.time_embed.transpose(1, 2)
                    new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                    new_time_embed = new_time_embed.transpose(1, 2)
                    x = x + new_time_embed
                else:
                    x = x + self.encoder.model.time_embed
                x = self.encoder.model.time_drop(x)
                x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
                x = torch.cat((cls_tokens, x), dim=1)
            
            output_cls   = [x[:, 0]]          # prima del primo block
            output_patch = [x[:, 1:]]         # shape [B, T*patches, D]
    
            ## Attention blocks
            for blk in self.encoder.model.blocks:
                x = blk(x, B, T, W)
                output_cls.append(x[:, 0])
                output_patch.append(x[:, 1:])
            num_register_tokens = 0  
        
        elif self.encoder.__class__.__name__.startswith("GroupChannelsVisionTransformer"): 
            b, C_all, H, W = image.shape
        
            x = image
            x_c_embed = []
            
            for i, group in enumerate(self.encoder.channel_groups):
                               
                x_c = x[:, group, :, :]
                x_c_embed.append(self.encoder.patch_embed[i](x_c))  # (N, L, D)
            

            x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
            _, G, L, D = x.shape
            
            # add channel embed
            channel_embed = self.encoder.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
            pos_embed = self.encoder.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)
    
            # Channel embed same across (x,y) position, and pos embed same across channel (c)
            #channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, c, L, cD)
            #channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1).contiguous()
            channel_embed = channel_embed.repeat(1, 1, pos_embed.shape[2], 1)


            pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, c, L, pD)
            pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)
    
            # add pos embed w/o cls token
            x = x + pos_channel  # (N, G, L, D)
            x = x.view(b, -1, D)  # (N, G*L, D)
    
            cls_pos_channel = torch.cat((self.encoder.pos_embed[:, :1, :], self.encoder.channel_cls_embed), dim=-1)  # (1, 1, D)
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = cls_pos_channel + self.encoder.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)
            x = self.encoder.pos_drop(x)
            
            output_cls   = [x[:, 0]]          # prima del primo block
            output_patch = [x[:, 1:]]
    
            for blk in self.encoder.blocks:
                x = blk(x)
                output_cls.append(x[:, 0])
                output_patch.append(x[:, 1:])
            num_register_tokens = 0
            
            # ─── dopo aver popolato output_cls / output_patch ──────────────────────────────
            G = len(self.encoder.channel_groups)          # 3 gruppi (RGB-VEG-GEO)
            
            for i in range(len(output_patch)):
                B, N, D = output_patch[i].shape           # N = G*L (es. 588)
                L = N // G                                # num patch per gruppo (196)
                output_patch[i] = (
                    output_patch[i]
                    .view(B, G, L, D)                     # (B, G, L, D)
                    .mean(dim=1)                          # media sui gruppi → (B, L, D)
                )

            # ora output_patch[i].shape[1] = 196  ✔️
            # ──────────────────────────────────────────────────────────────────────────────

    

            
        else:
            x, num_register_tokens = self.encoder.prepare_tokens_with_masks(image)
        
            output_cls = [x[:, 0, :]]
            output_patch = [x[:, 1 + num_register_tokens :, :]]
            
            for blk in self.encoder.blocks[0]: #QUI SAS
                x = blk(x)
                output_cls.append(x[:, 0, :])
                output_patch.append(x[:, 1 + num_register_tokens :, :])

        if self.encoder.__class__.__name__.startswith("TimeSformer"):  
            patch_tokens_split = None
            B, N, D = output_patch[-1].shape           # N = T * num_patches
            
        
            if self.strategy[0] == "split":
                #logger.info('faccio split')
                # [B, T, 196, D]
                num_patches = self.encoder.num_patches
                patch_tokens_split = output_patch[-1].reshape(B, T, num_patches, D)
            
            elif self.strategy[0] == "mean":
                #logger.info('faccio media')
                # Fai la media T?1 per TUTTI i livelli
                for i, p in enumerate(output_patch):
                    B, N, D = p.shape
                    num_patches = self.encoder.num_patches          # 196
                    T = N // num_patches                            # 3
                    output_patch[i] = p.reshape(B, T, num_patches, D).mean(dim=1)
            else:
                logger.info("nessuna strategia", self.strategy[0])
    
            out = self.lp(
                    output_cls,
                    output_patch,
                    patch_tokens_split=patch_tokens_split,
                    strategy=self.strategy[0],
            )
            
        elif self.encoder.__class__.__name__.startswith("DinoVisionTransformer"):
            out = self.lp(output_cls, output_patch)
        elif self.encoder.__class__.__name__.startswith("GroupChannelsVisionTransformer"):
            out = self.lp(output_cls, output_patch)
        else:
            raise ValueError(f"Nessun match con il nome dell'encoder")

        return out


class LP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dims: Dict[str, int],
        n_encoder_blocks: int,
        patch_tokens_split: Optional[torch.Tensor] = None,
        strategy: str = None,
        which_blocks: List[int] = None,
        hidden_dim: int = 768,
        last_hidden_dim: int = 3072,
        prenorm: bool = False,
        midnorm: bool = False,
        std: float = 0.02,
        use_only_last_layer: bool = False,
        loss: str = "cosine",
    ):
        super().__init__()


        #DA QUI NUOVO
        self.use_only_last_layer = use_only_last_layer
        

        self.n_encoder_blocks = n_encoder_blocks
        self.loss = loss
        
        
        # ──────────────────────────────────
        #  A) Modalità CROSS-ENTROPY
        # ──────────────────────────────────
        if loss == "cross-entropy":
            self.heads = nn.ModuleDict({
                hname: nn.ModuleDict({
                    "cls":  HeadLogits(input_dim, 19),
                    "patch": HeadLogits(input_dim, 19),
                })
                for hname in head_dims
            })
        #A QUI NUOVO

        else:
            if which_blocks is None:
                which_blocks = list(range(n_encoder_blocks))
            self.which_blocks = which_blocks

            def _make_head(output_dim):
                return nn.ModuleList(
                    [
                        (
                            AdaptMLP(
                                hidden_dim=(
                                    last_hidden_dim
                                    if bix == n_encoder_blocks - 1
                                    else hidden_dim
                                ),
                                prenorm=prenorm,
                                midnorm=midnorm,
                                dim=input_dim,
                                output_dim=output_dim,
                            )
                            if bix in which_blocks
                            else None
                        )
                        for bix in range(n_encoder_blocks)
                    ] 
                )
            self.heads = nn.ModuleDict(
                {
                    hname: nn.ModuleDict(
                        {
                            "cls": _make_head(head_dims[hname]),
                            "patch": _make_head(head_dims[hname]),
                        }
                    )
                    for hname in head_dims.keys()
                }
            )

            for m in self.heads.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=std)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x_cls:   list[torch.Tensor],     # len = n_blocks+1, ogni elem [B, D]
        x_patch: list[torch.Tensor],     # len = n_blocks+1, ogni elem [B, N, D]
        *,
        patch_tokens_split: torch.Tensor | None = None,  # [B, 3, N, D] se "split"
        strategy: str | None = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        out = defaultdict(dict)
        
        #DA QUI NUOVO
        # ------- 1) caso cross-entropy già gestito in __init__ ------------- #
        if self.loss == "cross-entropy":    # non arriverà qui: return anticipato
            raise RuntimeError("Should never reach")
        #A QUI NUOVO

        # ------- 2) iteriamo sulle head (A / B / C o mergedFeatures) -------- #
        for idx, (hname, head) in enumerate(self.heads.items()):
            xc, xp = 0, 0 


            #DA QUI NUOVO
            # --- (a) usa SOLO l’ultimo blocco --------------------------------
            if self.use_only_last_layer:
                bix = self.n_encoder_blocks - 1                 # ultimo block

                xc_in = x_cls[bix + 1]                          # [B, D]
                xp_in = (
                    patch_tokens_split[:, idx]                  # [B, N, D] split
                    if strategy == "split"
                    else x_patch[bix + 1]                       # [B, N, D]
                )

                xc = head["cls"][bix](xc_in)                    # [B, D]
                xp = head["patch"][bix](xp_in)                  # [B, …]
            #A QUI NUOVO

            # --- (b) somma / media sui blocchi in which_blocks --------------
            else:
                for bix in self.which_blocks:
                    xc = xc + head["cls"][bix](x_cls[bix + 1])
                    if strategy == "split":
                        if bix == self.which_blocks[-1]:         # solo sull’ultimo
                            xp = head["patch"][bix](patch_tokens_split[:, idx])
                    else:
                        xp = xp + head["patch"][bix](x_patch[bix + 1])

            out[hname]["cls"]   = xc                           
            out[hname]["patch"] = xp                           

        return out
        
        
class HeadLogits(nn.Module):
    def __init__(self, in_dim, out_dim, temp=0.04):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.temp = temp

    def forward(self, x):
        return self.fc(x) / self.temp


class AdaptMLP(nn.Module):

    def __init__(
        self,
        hidden_dim,
        prenorm=False,
        midnorm=False,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        scale=1.0,
        zinit=False,
        dim=None,
        output_dim=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prenorm = prenorm
        self.midnorm = midnorm
        self.norm_fn = norm_fn
        self.act_fn = act_fn
        self.scale = nn.Parameter(torch.ones(1).float()) if scale == 0.0 else scale
        self.zinit = zinit
        if dim is not None:
            self.setup(dim, output_dim)

    def extra_repr(self):
        repr = "scale={}, zinit={}".format(self.scale, self.zinit)
        return repr

    def setup(self, dim, output_dim=None):
        layers = []

        if self.prenorm:
            layers.append(self.norm_fn(dim))

        layers.append(nn.Linear(dim, self.hidden_dim))
        if self.zinit:
            nn.init.kaiming_uniform_(layers[-1].weight, a=math.sqrt(5))
            nn.init.zeros_(layers[-1].bias)

        if self.midnorm:
            layers.append(self.norm_fn(self.hidden_dim))

        layers.append(self.act_fn())

        layers.append(
            nn.Linear(self.hidden_dim, dim if output_dim is None else output_dim)
        )
        if self.zinit:
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.scale * self.layers(x)


def _build_encoder_from_args(args):

    if args.arch.startswith("vit"):
    
        encoder = vision_transformer.get_model(
            arch=args.arch,
            patch_size=args.patch_size,  # cos� compatibile con teacher
            img_size=args.image_size,
            in_chans=args.in_chans,
            drop_path_rate=args.drop_path_rate,
        )
                
        if getattr(args, "imagenet_pretrained", False):
            ckpt_path = IMAGENET_URLS["_".join(args.arch.split("_")[:2])]
            logger.info(f"Loading DINOv2 from {ckpt_path}")

            # ---------- 1) carica state_dict ----------
            state_dict = torch.load(ckpt_path, map_location="cpu")
            state_dict = state_dict["model"] if "model" in state_dict else state_dict

            # ---------- 2) rimuovi pos_embed ----------
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("pos_embed")}

            # ---------- 3) adatta conv 3->N canali ----------
            if args.in_chans != 3:
                w = state_dict["patch_embed.proj.weight"]  # [out, 3, kH, kW]

                # Nuovo tensor con N canali
                w_new = torch.zeros(w.size(0), args.in_chans, w.size(2), w.size(3))
                # Copia i pesi preaddestrati invertendo l'ordine RGB -> BGR
                w_new[:, :3, :, :] = w[:, [2, 1, 0], :, :]

                state_dict["patch_embed.proj.weight"] = w_new


            # ---------- 4) carica pesi ----------
            missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
            logger.info(f"? DINOv2 loaded | missing={len(missing)} unexpected={len(unexpected)}")
    
    elif args.arch.startswith("grouping"):
        print('encoder grouping')
        encoder = grouping.vit_base_patch16(
            patch_size=args.patch_size, img_size=args.image_size, in_chans=args.in_chans,
            channel_groups=[(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            num_classes=args.num_classes, drop_path_rate=args.drop_path_rate, global_pool=False,
        )
        
    else:
        encoder = timesformer.get_model(
            arch=args.arch,
            img_size=args.image_size,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
        )

    return encoder




    
    


def load_student_encoder_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), "Student checkpoint ({}) not found!".format(
        ckpt_fname
    )
    ckpt = torch.load(ckpt_fname, "cpu")

    encoder = _build_encoder_from_args(ckpt["args"])


    state_dict = ckpt.get(ckpt_key, ckpt)
    encoder.load_state_dict(
        {
            k.replace("module.", "").replace("encoder.", ""): v
            for k, v in state_dict.items()
            if "encoder." in k
        }
    )

    return encoder, ckpt["epoch"]
    
    

from typing import Dict, List
from collections import defaultdict
import torch
import torch.nn as nn

class IdentityLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dims: Dict[str, int],
        n_encoder_blocks: int,
        which_blocks: List[int] = None,
        hidden_dim: int = 768,
        last_hidden_dim: int = 3072,
        prenorm: bool = False,
        midnorm: bool = False,
        std: float = 0.02,
    ):
        super().__init__()
        if which_blocks is None:
            which_blocks = list(range(n_encoder_blocks))
        self.which_blocks = which_blocks
        self.head_dims = head_dims

        # crea un linear per ogni head_dim
        self.proj = nn.ModuleDict({
            hname: nn.Linear(input_dim, head_dims[hname])
            for hname in head_dims.keys()
        })

    def forward(
        self, x_cls: List[torch.Tensor], x_patch: List[torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        out = defaultdict(dict)
        for hname in self.head_dims.keys():
            out[hname]["cls"] = self.proj[hname](x_cls[-1])
            out[hname]["patch"] = self.proj[hname](x_patch[-1])
        return out



def build_student_from_args(args):

    encoder = _build_encoder_from_args(args)
    

    from teachers import TEACHER_CFG
    
    if "abf" in args.Teacher_strategy or "mean" in args.Teacher_strategy:
        logger.info("Adding mergedFeatures as head for concatenation ...")
        head_dims = {
            "mergedFeatures": max([TEACHER_CFG[tname]["num_features"] for tname in args.teachers])  # o metti a mano il valore corretto
        }
        #DA QUI NUOVO
        use_only_last_layer = False
        #A QUI NUOVO

    else:
        head_dims = {}
        for tname in args.teachers:
            #DA QUI NUOVO
            if "dino" in tname.lower():
                use_only_last_layer = True
                # aggiungi 3 teste per il teacher Dino
                for suffix in ["A", "B", "C"]:
                    head_name = f"{tname}_{suffix}"
                    '''
                    if args.loss == "cosine":
                        head_dims[head_name] = TEACHER_CFG[tname]["num_features"]
                    elif args.loss == "cross-entropy":
                        head_dims[head_name] = 19'''
                    head_dims[head_name] = TEACHER_CFG[tname]["num_features"]
            #A QUI NUOVO

            else:
                use_only_last_layer = False
                # una testa per ciascun teacher
                head_dims[tname] = TEACHER_CFG[tname.strip()]["num_features"]
   
    if encoder.__class__.__name__.startswith("GroupChannelsVisionTransformer"):
        logger.info('prendo solo la metà dei blocchi per fare lp')
        which_blocks = list(range(0, encoder.n_blocks, 2))  # un blocco sì, uno no
    else: 
        which_blocks = None

    
    if args.use_lp:
        lp_args = eval(args.lp_args)
        lp = LP(
            input_dim=encoder.embed_dim,
            head_dims=head_dims,
            loss=args.loss,
            use_only_last_layer=use_only_last_layer,
            n_encoder_blocks=encoder.n_blocks,
            which_blocks=which_blocks,
            **lp_args,
        )
    else:
      logger.info('Turning off LadderProjection, adding identity block ...')
      lp_args = eval(args.lp_args)
      lp = IdentityLP(
        input_dim=encoder.embed_dim,
        head_dims=head_dims,
        n_encoder_blocks=encoder.n_blocks,
        **lp_args,
      )
      

    model = UNIC(encoder, lp, args.in_chans, args.Student_strategy, args.num_frames)

    return model


def load_student_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), ckpt_fname
    ckpt = torch.load(ckpt_fname, "cpu")

    model = build_student_from_args(ckpt["args"])
    tnorms = ckpt["teacher_ft_stats"] if "teacher_ft_stats" in ckpt else None

    state_dict = ckpt.get(ckpt_key, ckpt)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})

    return model, tnorms, ckpt["epoch"]
