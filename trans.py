         
        rotation = transforms.RandomApply(
            [transforms.RandomRotation(degrees=180)],
            p=0.5  # probabilità di applicare la rotazione
        )
        
        augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            rotation,
            #transforms.Normalize(mean=mean, std=std),
        ])
        
        augmented_imagest = []
        augmented_imagess = []
        
        for img in image:
            augmented_imagest.append(augment(img))
            
        for img in image:
            augmented_imagess.append(augment(img))
            
        imaget = torch.stack(augmented_imagest)
        images = torch.stack(augmented_imagess)