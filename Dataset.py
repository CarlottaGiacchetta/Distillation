import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# Importa la classe BigEarthNet e il DataModule di base da torchgeo
from torchgeo.datasets import BigEarthNet
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms.transforms import AugmentationSequential
from torchgeo.transforms.color import RandomGrayscale
import kornia.augmentation as K

import itertools
from torchgeo.datasets import BigEarthNet
import torch
from kornia.augmentation import AugmentationBase2D

from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import SSL4EOS12
from pytorch_lightning import LightningDataModule


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
        if self._subset is not None and self._subset < len(lines):
            lines = random.sample(lines, self._subset)

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
            self.train_dataset = CustomBigEarthNet(
                split="train", 
                subset=self.subset,
                **self.kwargs
            )
            self.train_dataset.transform = self.transform
        if stage in ["fit", "validate", None]:
            self.val_dataset = CustomBigEarthNet(
                split="val", 
                subset=self.subset, 
                **self.kwargs
            )
        if stage in ["test", None]:
            self.test_dataset = CustomBigEarthNet(
                split="test", 
                subset=self.subset, 
                **self.kwargs
            )
            


class CustomSSL4EOS12DataModule(LightningDataModule):
    def __init__(self, val_split=0.2, seed=42,
                 batch_size=128, num_workers=4, **kwargs):
        super().__init__()
        self.val_split = val_split
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs          
    
    def setup(self, stage: str = None):
        full_ds = SSL4EOS12(**self.kwargs)
        val_len = int(len(full_ds) * self.val_split)
        train_len = len(full_ds) - val_len

        self.train_dataset, self.val_dataset = random_split(
            full_ds,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def _collate_season_flat(self, batch_list,
                             num_seasons: int = 4,
                             bands_per_season: int = 13,
                             drop_band_idx: int = 10):

        default_collate = torch.utils.data.default_collate
        batch = default_collate(batch_list)       

        x = batch["image"]                       
        B, C, H, W = x.shape
        assert C == num_seasons * bands_per_season

        
        chunks = torch.split(x, bands_per_season, dim=1)

       
        processed_chunks = [
            torch.cat([c[:, :drop_band_idx], c[:, drop_band_idx+1:]], dim=1) 
            for c in chunks
        ]
    
        batch["image"] = torch.cat(processed_chunks, dim=0)

        
        for k, v in batch.items():
            if k == "image":
                continue
            if torch.is_tensor(v):
                batch[k] = v.repeat_interleave(num_seasons, dim=0)
            elif isinstance(v, list):
                batch[k] = sum([v] * num_seasons, [])
            else:                                       
                batch[k] = [v_i for v_i in v for _ in range(num_seasons)]

        return batch           
        
                                
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_season_flat
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_season_flat
        )

        
        



def carica_dati(args, setup = "fit"):


    aa = True
    if aa:
        dm = CustomSSL4EOS12DataModule(
            root="/raid/home/rsde/cgiacchetta_unic/Distillation/data1/ssl4eo_s12",
            download=True,
            batch_size=int(args.batch_size/4),
            num_workers=args.num_workers,
            seasons=4,
            checksum=False
        )
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        validation_loader = dm.val_dataloader()
        return train_loader, validation_loader
    
    else:
    
        dm = CustomBigEarthNetDataModule(
                root=args.data_dir,
                download=True,  
                subset=None,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                bands=args.bands,     # 's1', 's2' oppure 'all'
                num_classes=args.num_classes,
                transform=None   # 19 o 43
            )
        print('dm', dm)
        
        dm.setup(setup)
        print('dm setup')
    
        if setup == "fit":
            train = dm.train_dataset
            train_loader = dm.train_dataloader()
            print('--creato train loader')
    
            validation = dm.val_dataset
            validation_loader = dm.val_dataloader()
            print('--creato validation loader')
            
            return train_loader, validation_loader
    
        else:
            test = dm.test_dataset
            test_loader = dm.test_dataloader()
            print('--creato test loader')
    
            return test_loader


