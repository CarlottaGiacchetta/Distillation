# fam.py
import torch
import torch.nn as nn
from typing import List, Dict
from collections import defaultdict


from dinov2.models.vision_transformer import vit_large

class AggregationLP(nn.Module):
    """
    Aggrega feature da RGB, VEG, GEO concatenandole lungo il canale
    e produce proiezioni separate per cls e patch token per ciascuna head.
    """
    def __init__(
        self,
        embed_dim: int,
        head_dims: Dict[str, int],
        use_attention: bool = False,
        std: float = 0.02
    ):
        super().__init__()
        self.use_attention = use_attention
        concat_dim = embed_dim * 3

        # Opzionale attenzione
        if use_attention:
            self.attn_cls = nn.MultiheadAttention(concat_dim, num_heads=8, batch_first=True)
            self.attn_patch = nn.MultiheadAttention(concat_dim, num_heads=8, batch_first=True)
        else:
            self.attn_cls = None
            self.attn_patch = None

        # Crea un MLP per ciascuna head e ciascun tipo di token
        self.heads = nn.ModuleDict({
            hname: nn.ModuleDict({
                "cls": nn.Linear(concat_dim, out_dim),
                "patch": nn.Linear(concat_dim, out_dim),
            })
            for hname, out_dim in head_dims.items()
        })

        # Inizializzazione pesi
        for m in self.heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x_cls_rgb: torch.Tensor,   # (B, D)
        x_cls_veg: torch.Tensor,   # (B, D)
        x_cls_geo: torch.Tensor,   # (B, D)
        x_patch_rgb: torch.Tensor, # (B, N, D)
        x_patch_veg: torch.Tensor, # (B, N, D)
        x_patch_geo: torch.Tensor, # (B, N, D)
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        out = defaultdict(dict)

        # Concatenazione cls e patch token lungo l'ultima dimensione
        x_cls = torch.cat([x_cls_rgb, x_cls_veg, x_cls_geo], dim=-1)   # (B,1,3D)
        x_patch = torch.cat([x_patch_rgb, x_patch_veg, x_patch_geo], dim=-1)        # (B,N,3D)

        # Attenzione opzionale
        if self.attn_cls is not None:
            x_cls, _ = self.attn_cls(x_cls, x_cls, x_cls)   # (B,1,3D)
        if self.attn_patch is not None:
            x_patch, _ = self.attn_patch(x_patch, x_patch, x_patch)


        # Proietta ogni testa
        for hname, head in self.heads.items():
            out[hname]["cls"] = head["cls"](x_cls)               # (B, out_dim)
            # Pooling sulle patch token (media)
            projected_patch = head["patch"](x_patch)  # (B, N, D)
            out[hname]["patch"] = projected_patch

        return out




class DinoV2LargeThreeFAM(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder = encoder 
        from modeling.unic import LP
        self.agg_lp = LP(
            input_dim=self.encoder.embed_dim,
            head_dims={
                "A": embed_dim,
                "B": embed_dim,
                "C": embed_dim
            },
            loss="cosine",
            use_only_last_layer=True,
            n_encoder_blocks=self.encoder.n_blocks,
        )              
        
        '''self.agg_lp = AggregationLP(
            embed_dim=embed_dim,
            head_dims={
                "A": embed_dim,
                "B": embed_dim,
                "C": embed_dim
            },
            use_attention=False,  # O True se vuoi
            std=0.02
        )
        '''

        # opzionale: congela backbone
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _token_features(self, x, modname):    
        feats = self.encoder.forward_features(x,modname=modname)
        feats["x_norm_clstoken"] = feats["x_norm_clstoken"][-1]       # ? solo l'ultimo CLS token
        feats["x_norm_patchtokens"] = feats["x_norm_patchtokens"][-1] # ? solo l'ultimo PATCH token
        return feats    # [B, N, D]

    def forward_features(self, img, teacher=True):
        
        if not hasattr(self, 'patch_embed_rgb'):
            feats_avg = self.encoder.forward_features(img)            
            
        else:
            feats_rgb = self._token_features(img[:,0:3], modname="rgb")
            feats_veg = self._token_features(img[:,3:6], modname="veg")
            feats_geo = self._token_features(img[:,6:9], modname="geo")

            feats_all = {}
            feats_avg = {}
    
            feats_avg["x_norm_clstoken"] = (
                feats_rgb["x_norm_clstoken"]
                + feats_veg["x_norm_clstoken"]
                + feats_geo["x_norm_clstoken"]
            ) / 3
            
            feats_avg["x_norm_patchtokens"] = (
                feats_rgb["x_norm_patchtokens"]
                + feats_veg["x_norm_patchtokens"]
                + feats_geo["x_norm_patchtokens"]
            ) / 3
            
        
        # --- dopo aver calcolato feats_avg -----------------------------
        x_cls_list = [None] + [feats_avg["x_norm_clstoken"]] * self.encoder.n_blocks
        x_patch_list = [None] + [feats_avg["x_norm_patchtokens"]] * self.encoder.n_blocks
        
        out = self.agg_lp(x_cls_list, x_patch_list)               # ora LP riceve 3-D

        
        '''
        
        out = self.agg_lp(
            feats_rgb["x_norm_clstoken"],
            feats_veg["x_norm_clstoken"],
            feats_geo["x_norm_clstoken"],
            feats_rgb["x_norm_patchtokens"],
            feats_veg["x_norm_patchtokens"],
            feats_geo["x_norm_patchtokens"],
        )'''

        return out
        
        
    def forward(self, img):
        return self.forward_features(img, teacher=False)
        
        
        
def build_dinov2large_with_fams(chekpoint="", loss = "cosine"):
    backbone = vit_large(
        patch_size=14,
        teacher=True,
        in_chans=9
    )
    if loss == "cosine":
        embed_dim=1024
    elif loss == "cross-entropy":
        embed_dim=19
    else: 
        print('NON CONOSCO LA LOSSS')

    model = DinoV2LargeThreeFAM(
        backbone=backbone,
        embed_dim=embed_dim,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def build_dinov2large_baseline(chekpoint="", loss = "cosine"):
    backbone = vit_large(
        patch_size=14,
        teacher=False,
        in_chans=9
    )
    if loss == "cosine":
        embed_dim=1024
    elif loss == "cross-entropy":
        embed_dim=19
    else: 
        print('NON CONOSCO LA LOSSS')

    model = DinoV2LargeThreeFAM(
        encoder=backbone,
        embed_dim=embed_dim,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model