import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Importa la classe BigEarthNet e il DataModule di base da torchgeo
from torchgeo.datasets import BigEarthNet
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms.transforms import AugmentationSequential, _RandomNCrop, _Clamp
from kornia.augmentation import Resize
from torchgeo.transforms.color import RandomGrayscale
import kornia.augmentation as K
import torch.nn.functional as F

import itertools
from torchgeo.datasets import BigEarthNet
import torch
from kornia.augmentation import AugmentationBase2D


class CustomLambda(AugmentationBase2D):
    def __init__(self, fn):
        super().__init__(p=1.0, same_on_batch=False, keepdim=True)
        self.fn = fn

    def apply_transform(self, input, params, flags, transform):
        return self.fn(input)


class CustomBigEarthNet(BigEarthNet):
    def __init__(self, subset: int = None, *args, **kwargs):
        self._subset = subset  # Salva il valore del subset prima di inizializzare la classe base
        super().__init__(*args, **kwargs)

    def _load_folders(self) -> list[dict[str, str]]:
        
        filename = self.splits_metadata[self.split]['filename']
        print(filename)
        dir_s1 = self.metadata['s1']['directory']
        dir_s2 = self.metadata['s2']['directory']

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            


        # Applica subito il subset (se richiesto)
        if self._subset is not None:
            lines = lines[500:500+self._subset]

        pairs = [line.split(',') for line in lines]

        folders = [
            {
                's1': os.path.join(self.root, dir_s1, pair[1]),
                's2': os.path.join(self.root, dir_s2, pair[0]),
            }
            for pair in pairs
        ]
        return folders


class CustomBigEarthNetDataModule(NonGeoDataModule):
    def __init__(self, subset: int = None, transform=None, batch_size: int = 64, num_workers: int = 0, **kwargs):
        """
        Args:
            subset (int, optional): Numero di campioni da usare.
            transform: Pipeline di trasformazioni da applicare ai dati.
            batch_size (int): Dimensione del batch.
            num_workers (int): Numero di processi per il DataLoader.
            **kwargs: Altri parametri da passare al dataset.
        """
        
        self.subset = subset
        self.transform = transform
        self.kwargs = kwargs  
        super().__init__(CustomBigEarthNet, batch_size, num_workers, **kwargs)

    def setup(self, stage: str = None):
        if stage in ["fit", None]:
            self.train_dataset = CustomBigEarthNet(split="train", subset=self.subset, **self.kwargs)
            self.train_dataset.transforms = self.transform
        if stage in ["fit", "validate", None]:
            self.val_dataset = CustomBigEarthNet(split="val", subset=self.subset, **self.kwargs)
            self.val_dataset.transforms = self.transform
        if stage in ["test", None]:
            self.test_dataset = CustomBigEarthNet(split="test", subset=self.subset, **self.kwargs)
            self.test_dataset.transforms = self.transform

def min_max_fn(x):
    min_val = x.amin(dim=(2, 3), keepdim=True)
    max_val = x.amax(dim=(2, 3), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-8)

   

class ApplyToSingleImage(torch.nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            K.Resize((img_size, img_size), p=1.0, keepdim=False),
            K.RandomHorizontalFlip(p=0.5, keepdim=False),
            K.RandomVerticalFlip(p=0.5, keepdim=False),
            K.RandomRotation(degrees=90.0, p=0.5, keepdim=False),
        )

    def forward(self, sample):
        img = sample["image"].unsqueeze(0)  # da [C, H, W] ? [1, C, H, W]
        img = min_max_fn(img)
        img = self.transforms(img)
        sample["image"] = img.squeeze(0)
        return sample

                      
def get_transforms(img_size):
    return ApplyToSingleImage(img_size)





def carica_dati(args, setup="fit"):
    
    if args.transform:
        print("args.transform =", args.transform)
        train_transform = get_transforms(args.image_size)
    else:
        train_transform = AugmentationSequential([
            K.Resize((args.image_size, args.image_size), p=1.0, align_corners=False)
        ], data_keys=["image"])

    dm = CustomBigEarthNetDataModule(
        root=args.data_dir,
        download=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bands=args.bands,
        num_classes=args.num_classes,
        transform=train_transform,
    )

    dm.setup(setup)

    if setup == "fit":
        return dm.train_dataloader(), dm.val_dataloader()
    else:
        return dm.test_dataloader()


