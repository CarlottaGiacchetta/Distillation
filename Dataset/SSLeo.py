import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from torch.utils.data import Subset
from tqdm import tqdm

# Importa la classe BigEarthNet e il DataModule di base da torchgeo
from torchgeo.datasets import BigEarthNet
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms.transforms import AugmentationSequential
from torchgeo.transforms.color import RandomGrayscale
import kornia.augmentation as K
from torch.utils.data._utils.collate import default_collate

import itertools
from torchgeo.datasets import BigEarthNet
import torch
from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import SSL4EOS12
from pytorch_lightning import LightningDataModule
import random
from torch.utils.data import Dataset

class SafeDataset(Dataset):
    """
    Avvolge un dataset e, se un file e' mancante o corrotto,
    riprova con un indice diverso; dopo `max_retry` ritorna None.
    """
    def __init__(self, dataset, max_retry: int = 10):
        self.ds = dataset
        self.max_retry = max_retry

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        for _ in range(self.max_retry):
            try:
                return self.ds[idx]
            except FileNotFoundError:
                idx = random.randint(0, len(self.ds) - 1)
        return None          # sarà filtrato dal collate




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
        full_ds = SafeDataset(SSL4EOS12(**self.kwargs))
        # n-val e n-train identici a prima
        val_len   = int(len(full_ds) * self.val_split)
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
    
        # ?? 1) scarta eventuali None ritornati da SafeDataset
        batch_list = [b for b in batch_list if b is not None]
        if not batch_list:              # capita di rado; DataLoader riprova
            return None
    
        # ?? 2) tutto il resto resta invariato
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
    
        # duplichiamo meta-dati per tutte le stagioni
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

        
        



def carica_ssl(args, setup = "fit"):

    dm = CustomSSL4EOS12DataModule(
        root=args.data_dir,
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

    


