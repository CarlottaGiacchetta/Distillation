#from .vit_dino import dino_vitbase
#from .vit_deit3 import deit3_vitbase
#from .vit_dbotft import dbotft_vitbase
from functools import partial

from teachers.ScaleMae import scalemae_RGB, scalemae_VEG, scalemae_GEO
from dinov2.models.aggregation import build_dinov2large_with_fams
from dinov2.models.vision_transformer import vit_large
#from teachers.ViT import ViT, ViT_RGB, ViT_VEG, ViT_GEO, ViT_large

TEACHER_CFG = {
    "scalemae_rgb": {
        "loader": scalemae_RGB,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/Distillation/modell/scalemae_RGB/best-checkpoint.ckpt", #"/workspace/models/scalemae_RGB/best-checkpoint.ckpt"
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        "finetuning_bands": "rgb"
        
    },
    "scalemae_veg": {
        "loader": scalemae_VEG,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/Distillation/modell/scalemae_VEG/best-checkpoint.ckpt",
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        "finetuning_bands": "veg"
        
    },
    "scalemae_geo": {
        "loader": scalemae_GEO,
        "ckpt_path": "/raid/home/rsde/cgiacchetta_unic/Distillation/modell/scalemae_GEO/best-checkpoint.ckpt", 
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        "finetuning_bands": "geo" 
    },

    "DinoV2Large": {
        "loader": build_dinov2large_with_fams,  
        "ckpt_path": "", 
        "ckpt_key": "model",
        "num_features": 1024,
        "resolution": 224,
        "finetuning_bands": "nove" 
    },

}