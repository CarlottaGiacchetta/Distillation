import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy


import argparse
from teachers.config import CONFIG
from dinov2.models.grouping import vit_base_patch16, vit_large_patch16


from torchmetrics.classification import MultilabelAveragePrecision

class GroupIng(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # Supporta sia Namespace che dict, e fallback se args è None
        self.save_hyperparameters()  # logs hyperparameters for reproducibility
    
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        '''  
        self.encoder = vit_base_patch16(
            patch_size=args.get("patch_size", 16), img_size=args.get("image_size", 224), in_chans=args.get("in_chans", 12),
            channel_groups=[(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            num_classes=args.get("num_classes", 19), drop_path_rate=args.get("drop_path_rate", 0.0), global_pool=False,
        )
        '''
        self.encoder = vit_large_patch16(
            patch_size = args.get("patch_size", 16), img_size=args.get("image_size", 224), in_chans=args.get("in_chans", 12),
            channel_groups=[(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            num_classes=args.get("num_classes", 19), drop_path_rate=args.get("drop_path_rate", 0.0), global_pool=True,
        )
        #'''
        
        self.num_classes = args.get("num_classes", 19)
        self.finetune_backbone = args.get("finetune_backbone")
        self.classifier = nn.Linear(self.encoder.embed_dim, self.num_classes)
        
        self.finetuning_bands = "all"
        self.bands = CONFIG[self.finetuning_bands]["bands"]
        self.mean = CONFIG[self.finetuning_bands]["mean"]
        self.std = CONFIG[self.finetuning_bands]["std"]

        checkpoint_path = args.get("checkpoint_path")
        if checkpoint_path:
            state = torch.load(checkpoint_path)

            # Cerca i pesi nel posto giusto
            if isinstance(state, dict):
                if "model" in state:
                    state_dict = state["model"]
                elif "state_dict" in state:
                    state_dict = state["state_dict"]
                else:
                    state_dict = state
            else:
                state_dict = state  # fallback
            
            # Filtra solo i pesi del ViT encoder
            encoder_state_dict = {
                k.replace("module.encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("module.encoder.")
            }

            # Caricamento con tolleranza
            missing, unexpected = self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"[INFO] Loaded encoder from {checkpoint_path}")
            print(f"[INFO] Missing keys: {missing}")
            print(f"[INFO] Unexpected keys: {unexpected}")

        if not args.get("finetune_backbone"):
            print('encoder freezato')
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.lr = args.get("lr", 1e-3)
        self.wd = args.get("wd", 1e-4)
        self.metric = MultilabelAveragePrecision(num_labels=self.num_classes)
        self.image_size = args.get("image_size", 224)
        

        

    def forward(self, x):
        #x = x[:, self.bands, :, :]
        _, _, H, W = x.shape
        if (H, W) != (self.image_size, self.image_size):
            #print('faccio resize delle immagini perchè ho dimensioni: ', (H, W))
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False) # x: (B, 3, 120, 120) ? (B, 3, 224, 224)
        #x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        features = self.encoder.forward_features(x)
        cls_token = features
        print(cls_token.shape)
        return self.classifier(cls_token)
    

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("train_loss", loss, prog_bar=True)
        self.log("logits_mean", logits.mean(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("logits_std", logits.std(), prog_bar=True, on_step=False, on_epoch=True)
        
        self.log("train_loss", loss, prog_bar=True)
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
    
    def configure_optimizers(self):
    
        if not self.finetune_backbone:
            print('encoder freezato')
            param_groups = [{"params": self.classifier.parameters(), "lr": self.lr}]
        else:
            param_groups = [
                {"params": self.encoder.parameters(), "lr": self.lr * 0.1},
                {"params": self.classifier.parameters(), "lr": self.lr}
            ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.wd)

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





