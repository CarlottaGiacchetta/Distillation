import os
import random
from torchgeo.datasets import BigEarthNet
from torchgeo.datamodules import NonGeoDataModule



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


def carica_bigearthnet(args, setup = "fit"):

    print(args)

    dm = CustomBigEarthNetDataModule(
            root=args.data_dir,
            download=True,  
            subset=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            bands=args.bands,     # 's1', 's2' oppure 'all'
            num_classes=args.num_classes,
        )
    print('dm', dm)
    
    dm.setup(setup)
    print('dm setup')

    if setup == "fit":
        train_loader = dm.train_dataloader()
        print('--creato train loader')

        validation_loader = dm.val_dataloader()
        print('--creato validation loader')
        
        return train_loader, validation_loader

    else:
        test_loader = dm.test_dataloader()
        print('--creato test loader')

        return test_loader


