from torchgeo.datamodules.ssl4eo import SSL4EOS12DataModule
import os

def carica_dati(args, setup = "fit"):

    dm = SSL4EOS12DataModule(
        root=os.path.join(args.data_dir, "ssl4eos12"),
        seasons=4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checksum=True
    )
    
    dm.setup(stage=setup)

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

    
    