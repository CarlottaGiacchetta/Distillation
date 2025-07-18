import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics.classification import MultilabelAveragePrecision
#from torchgeo.models.scale_mae import scalemae_large_patch16, ScaleMAELarge16_Weights
from torchgeo.models import scale_mae
from teachers.config import CONFIG
from typing import Any
from functools import partial




class ScaleMAE(pl.LightningModule):

    def __init__(self, args=None):
        super().__init__()

        # Supporta sia Namespace che dict, e fallback se args è None
        args = vars(args) if isinstance(args, argparse.Namespace) else args or {}

        # Parametri con valori di default per test
        self.lr = args.get("lr", 1e-3)
        self.wd = args.get("wd", 1e-4)
        self.image_size = args.get("image_size", 224)
        self.num_classes = args.get("num_classes", 19)
        self.use_weight = args.get("use_weight", False)
        self.finetuning_bands = args.get("finetuning_bands", "rgb")
        self.concat = args.get("concat", False)

        self.save_hyperparameters()  # salva quelli passati

        #self.backbone = scalemae_tiny_patch14()
        self.backbone = scale_mae.scalemae_large_patch16()
        self.classifier = nn.Linear(self.backbone.embed_dim, self.num_classes)

        # Metriche
        self.metric = MultilabelAveragePrecision(num_labels=self.num_classes)

        # Bande e normalizzazione
        self.bands = CONFIG[self.finetuning_bands]["bands"]
        self.mean = CONFIG[self.finetuning_bands]["mean"]
        self.std = CONFIG[self.finetuning_bands]["std"]
       
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.backbone.parameters(), "lr": self.lr * 0.1},
            {"params": self.classifier.parameters(), "lr": self.lr}
        ], weight_decay=self.wd)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                verbose=True
            ),
            "monitor": "val_map",
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

    def forward(self, x):
        x = x[:, self.bands, :, :] # x: (B, 12, H, W) → (B, 3, H, W)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False) # x: (B, 3, 120, 120) → (B, 3, 224, 224)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        features = self.backbone.forward_features(x) # (B, 197, D)
        cls_token = features[:, 0, :]  # (B, D)
        return self.classifier(cls_token)  # (B, num_classes)

    
    def forward_features(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        
        features = self.backbone.forward_features(x)  # (B, 1+N, D)

        
        
        cls_token = features[:, 0]       # (B, D)
        patch_tokens = features[:, 1:]   # (B, N, D)
        gp = patch_tokens.mean(dim=1)    # global pooled features (B, D)
        
        return {
            "x_norm_clstoken": gp,                   # token globale (può anche essere cls_token)
            "x_norm_patchtokens": patch_tokens,      # patch tokens finali
            "x_prenorm": features,                   # tutti i token
            "x_prenorm_clstoken": cls_token,         # cls token
            "x_prenorm_patchtokens": patch_tokens,   # patch tokens
        }
    

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        preds = torch.sigmoid(logits)
        self.metric.update(preds, y.int())
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_map = self.metric.compute()
        self.log("val_map", val_map, prog_bar=True)
        self.metric.reset()

    def infer(self, batch, threshold=0.5):
        self.eval()
        with torch.no_grad():
            x = batch["image"]
            x = x[:, self.rgb_band_indices, :, :]
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            logits = self(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
        return preds



def scalemae_tiny_patch14( *args: Any, **kwargs: Any
) -> scale_mae.ScaleMAE:
    
    model = scale_mae.ScaleMAE(
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        *args,
        **kwargs,
    )

    return model


def scalemae_RGB(checkpoint_path, loss=None):
    model = ScaleMAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

def scalemae_VEG(checkpoint_path, loss=None):
    model = ScaleMAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model
    
def scalemae_GEO(checkpoint_path, loss=None):
    model = ScaleMAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model