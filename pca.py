import argparse
import os
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from teachers.teachers_config import TEACHER_CFG
from teachers.config import CONFIG
from teachers.ViT import ViT
from Dataset import carica_dati
from dinov2.models import vision_transformer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch",
        type=str,
        default="vit_base", #or time_tiny
        help="Architecture of the student model. "
        "See dinov2/models/vision_transformer.py for options. See dinov2/models/timesformer.py for options.",
    )
    
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size for the student model.",
    )
    
    parser.add_argument(
        "--teachers",
        type=str,
        default="scalemae_rgb,scalemae_veg,scalemae_geo",#, scalemae_veg",
        help="Comma-separated list of teacher names.",
    )
    
    parser.add_argument(
        "--student",
        type=str,
        default="",#, scalemae_veg",
        help="chpt dello student",
    )
    
    parser.add_argument(
        "--Teacher_strategy",
        type=lambda s: eval(s) if s else [],  # converte stringa in lista
        default=[],
        help='Fusion strategy, e.g., \'["rab", "abf"]\''
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path for the BigearthNet data directory",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (for both training and validation). "
        "We assume the input images are square.",
    )
    

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )

    args = parser.parse_args()

    args.teachers = args.teachers.split(",")
    args.num_cpus = len(os.sched_getaffinity(0))
    args.transform = False
    args.batch_size = 128
    args.num_workers = 4
    args.bands = 's2'
    args.num_classes = 19
    args.in_chans = 9
    args.checkpoint_path = args.student
    
    os.makedirs(args.output_dir, exist_ok=True)
    

    return args
    
    
    
def build_teachers(teacher_names):
    teachers = {}
    for tname in teacher_names:
        tname = tname.strip()
        print("Loading teacher '{}'".format(tname))
        
        
        cfg        = TEACHER_CFG[tname]
        ckpt_path  = cfg.get("ckpt_path", "")
        ckpt_key   = cfg.get("ckpt_key", None)
        loader_fn  = cfg["loader"]   
        
         
        model = loader_fn(ckpt_path)
    
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"Loading checkpoint for teacher '{tname}' from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd   = ckpt[ckpt_key] if ckpt_key in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
        else:
            if ckpt_path:
                print(
                    f"Checkpoint for teacher '{tname}' not found at '{ckpt_path}'. "
                    "The model will start with random weights."
                )
    
        for param in model.parameters():
            param.requires_grad = False
    
        model = model.cuda()
        teachers[tname] = model
    return teachers
    
def normalize_min_max(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor.clamp(0, 1)
    return tensor  

def main(args):


    n_image = 100

    test_loader = carica_dati(args, setup = 'test')
 
    sample = next(iter(test_loader))
    print(f"Shape batch immagini: {sample['image'].shape}")


   
    print("Loading teachers ...")
    teachers = build_teachers(args.teachers)
    
    
    
    image = sample["image"] 
    label = sample["label"]
    image = image.cuda(non_blocking=True)
    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
       
    
    
    output = {}
    student = ViT(args)
    
    for it, sample in enumerate(test_loader):
        image = sample["image"] 
        label = sample["label"]
        image = image.cuda(non_blocking=True)
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    
        
       
        for tname in teachers.keys():
            image_copy = image
            finetuning_bands = TEACHER_CFG[tname]["finetuning_bands"]
               
            device = "cuda"
            mean = CONFIG[finetuning_bands]["mean"].to(device)
            std = CONFIG[finetuning_bands]["std"].to(device)
            bands = CONFIG[finetuning_bands]["bands"]
            image_copy = image_copy[:, bands, :, :]
            image_copy = (image_copy - mean) / std
            
            output[tname] = teachers[tname].backbone.forward_features(image_copy)[n_image][1:] #prendo solo prima immagine e tolgo cls
            
        
        image_copy = image
        finetuning_bands = "nove"
        device = "cuda"
        mean = CONFIG[finetuning_bands]["mean"].to(device)
        std = CONFIG[finetuning_bands]["std"].to(device)
        bands = CONFIG[finetuning_bands]["bands"]
        image_copy = image_copy[:, bands, :, :]
        image_copy = (image_copy - mean) / std
        
        student.encoder = student.encoder.cuda()
        f = student.encoder.forward_features(image_copy)
        output['student'] = f["x_norm_patchtokens"][n_image]

        
        # Numero totale di plot = 1 (RGB) + N teacher + 1 student
        ncols = len(output) + 1
        fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4), dpi=200, constrained_layout=True)
        
        # --- Immagine RGB originale
        rgb_img = normalize_min_max(image[:, [3, 2, 1], :, :][n_image].permute(1, 2, 0).detach().cpu())
        axs[0].imshow(rgb_img)
        axs[0].set_title("RGB")
        axs[0].axis("off")
        
        # --- PCA di ciascun modello
        for i, name in enumerate(output.keys(), start=1):
            feat = output[name]  # [196, D] or [N, D]
            pca = PCA(n_components=3)
            patch_pca = torch.from_numpy(
                pca.fit_transform(feat.detach().cpu().numpy())
            )
            patch_dim = int(np.sqrt(feat.shape[0]))
            patch_pca = patch_pca.reshape([patch_dim, patch_dim, 3]).permute(2, 0, 1)
            patch_pca = (
                torch.nn.functional.interpolate(patch_pca.unsqueeze(0), 224, mode="nearest")
                .squeeze(0)
                .permute(1, 2, 0)
            )
            patch_pca = normalize_min_max(patch_pca)
            
            axs[i].imshow(patch_pca)
            axs[i].set_title(name)
            axs[i].axis("off")
        
        # --- Salva figura completa
        out_path = os.path.join(args.output_dir, f"pca_grid_{it}.png")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Salvato plot con PCA in {out_path}_{it}")
        plt.close(fig) 


if __name__ == "__main__":
    args = get_args()
    main(args)