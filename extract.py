from torchgeo.datasets import SSL4EOS12
from torch.utils.data import DataLoader, random_split
from torchgeo.datamodules import SSL4EOS12DataModule
import torch


root = "/raid/home/rsde/cgiacchetta_unic/Distillation/data1/ssl4eo_s12"

'''
dataset = SSL4EOS12(
    root=root,
    split="s2c",       
    seasons=4,          
    download=True,      
    checksum=False      
)


loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)'''

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import SSL4EOS12
import torch


from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import SSL4EOS12
from pytorch_lightning import LightningDataModule
import torch

from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import SSL4EOS12
from pytorch_lightning import LightningDataModule
import torch

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

        
        
dm = CustomSSL4EOS12DataModule(
    root="/raid/home/rsde/cgiacchetta_unic/Distillation/data1/ssl4eo_s12",
    download=True,
    batch_size=128,
    num_workers=4,
    seasons=4,
    checksum=False
)

dm.setup("fit")

train_loader = dm.train_dataloader()
validation_loader = dm.val_dataloader()

print("=== Summary DataModule ===")
print(dm)

print("\n=== Summary train_loader ===")
print("Oggetto:", train_loader)
print("Numero di batch:", len(train_loader))
print("Batch size:", train_loader.batch_size)
print("Dataset di train:", train_loader.dataset)
print("Numero di esempi:", len(train_loader.dataset))

print("Oggetto:", train_loader)
print("Numero di batch:", len(validation_loader))
print("Batch size:", validation_loader.batch_size)
print("Dataset di train:", validation_loader.dataset)
print("Numero di esempi:", len(validation_loader.dataset))

# Se vuoi le bande:
if hasattr(train_loader.dataset, "bands"):
    print("Bande:", train_loader.dataset.bands)
    
batch = next(iter(train_loader))

print(batch.keys())
print(batch['image'].shape)




