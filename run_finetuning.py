import argparse
import os
import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from teachers.ScaleMae import ScaleMAE
from Dataset import carica_dati
from utils import print_program_info

logger = logging.getLogger()


def get_args():
    parser = argparse.ArgumentParser()

    # === Dataset and input options ===
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/tesi_carlotta/data",
        help="Path to the dataset directory (default: local path).",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="s2",
        help="Multispectral bands of BigEarthNet.",
    )
    parser.add_argument(
        "--fintuning_bands",
        type=str,
        default="rgb",  # Options: rgb, vegetation, rocks
        help="Subset of bands to use for fine-tuning.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size for training and validation (assumes square images).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=19,
        help="Number of target classes for BigEarthNet classification.",
    )
    
    
    # Architecture name, drop_path rate, patch_size etc. can also be set for ViT.
    parser.add_argument(
        "--arch",
        type=str,
        default="scalemae_large",
        help="ViT architecture to use when selecting the ViT baseline.",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Patch size to use for the vision transformer (ViT).",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.0,
        help="Drop path rate for the vision transformer (ViT).",
    )

    # === Checkpoint directory and training hyperparameters ===
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="D:/tesi_carlotta/checkpoints/vit",
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="D:/tesi_carlotta/checkpoints/vit",
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used during fine-tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )

    # === Optimization ===
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=5e-3,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Weight decay for optimizer.",
    )

    # === Augmentations and other options ===
    parser.add_argument(
        "--transform",
        type=bool,
        default=True,
        help="Whether to apply data augmentations.",
    )
    

    # === Output ===
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save logs and checkpoints.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def main(args):
    print_program_info(args)
    train_loader, validation_loader = carica_dati(args)

    model = ScaleMAE(args)

    logger_tb = TensorBoardLogger(os.path.join(args.output_dir, 'tb_logs'), name=f"scalemae_Finetuner")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,  
        monitor="val_map",              
        mode="max",                     
        save_top_k=1,                   
        filename="best-checkpoint",    
        verbose=True
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_map",
        patience=args.patience,
        mode="max",
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # use GPU if available
        logger=logger_tb,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)


if __name__ == "__main__":
    args = get_args()
    main(args)
