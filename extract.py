from torchgeo.datasets import SSL4EOS12
from torch.utils.data import DataLoader

# Percorso di destinazione per il dataset
root = "/raid/home/rsde/cgiacchetta_unic/Distillation/data1/ssl4eo_s12"

# Istanzia il dataset con download automatico
dataset = SSL4EOS12(
    root=root,
    split="s2c",        # Altri possibili: "s1", "s2a"
    seasons=4,          # Da 1 a 4
    download=True,      # Scarica automaticamente da HuggingFace
    checksum=False      # Evita controllo MD5 se vuoi velocizzare
)

# DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Esempio: stampa una batch
for batch in loader:
    print("Image shape:", batch["image"].shape)
    print("Coordinates:", batch["x"], batch["y"])
    print("Timestamp:", batch["t"])
    break
