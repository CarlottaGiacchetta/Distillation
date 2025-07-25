import torch

CONFIG = {
            "rgb": {
                "bands": [3, 2, 1],
                "mean": torch.tensor([588.41, 614.06, 438.3721]).view(1, 3, 1, 1),
                "std": torch.tensor([684.57, 603.30, 607.0269]).view(1, 3, 1, 1),
            },
            "veg": {
                "bands": [4, 5, 6],
                "mean": torch.tensor([942.75, 1769.85, 2049.48]).view(1, 3, 1, 1),
                "std": torch.tensor([727.58, 1087.43, 1261.43]).view(1, 3, 1, 1),
            },
            "geo": {
                "bands": [7, 10, 11],
                "mean": torch.tensor([2193.2919921875, 1568.2117919921875, 997.715087890625]).view(1, 3, 1, 1),
                "std": torch.tensor([1369.3717041015625, 1063.9197998046875, 806.8846435546875]).view(1, 3, 1, 1),
            },
            "mix": {
                "bands": [3, 5, 10],
                "mean": torch.tensor([588.41, 1769.85, 1568.22]).view(1, 3, 1, 1),
                "std": torch.tensor([684.57, 1087.43, 1063.92]).view(1, 3, 1, 1),
            },
            "all": {
                "bands": [0,1,2,3,4,5,6,7,8,9,10,11],
                "mean": torch.tensor([360.6375, 438.3721, 614.0557, 588.4096, 942.7473, 1769.8485, 2049.4758, 2193.2920, 2235.4866, 2241.0911, 1568.2118, 997.7151]).view(1, 12, 1, 1),
                "std": torch.tensor([563.1734, 607.0269, 603.2968, 684.5688, 727.5784, 1087.4288, 1261.4303, 1369.3717, 1342.4904, 1294.3555, 1063.9198, 806.8846]).view(1, 12, 1, 1),
            },
            "nove": {
                "bands": [1,2,3,4,5,6,7,10,11],
                "mean": torch.tensor([438.3721, 614.0557, 588.4096, 942.7473, 1769.8485, 2049.4758, 2193.2920, 1568.2118, 997.7151]).view(1, 9, 1, 1),
                "std": torch.tensor([607.0269, 603.2968, 684.5688, 727.5784, 1087.4288, 1261.4303, 1369.3717, 1063.9198, 806.8846]).view(1, 9, 1, 1),
            },
            }
            
