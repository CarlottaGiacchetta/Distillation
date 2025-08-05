from torchgeo.datamodules import BigEarthNetDataModule
import os

def carica_dati(args, setup = "fit"):

    dm = BigEarthNetDataModule(
        root=os.path.join(args.data_dir, "bigearthnet"),
        bands="s2",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
        checksum=True
    )
    dm.prepare_data()
    dm.setup()
    
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
    



root=args.data_dir,
        split="s2a",
        seasons=4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
        checksum=True