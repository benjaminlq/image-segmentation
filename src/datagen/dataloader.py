from torch.utils.data import DataLoader
from torchvision import transforms
import os
import config
from datagen.datasets import StanfordDataset
from torchvision.datasets import VOCSegmentation
import h5py

class StanfordDataLoader(DataLoader):
    def __init__(
        self, 
        data_path: str = os.path.join(config.DATA_DIR, "dataset.hdf5"),
        augmentation: bool = False,
        val_ratio: float = 0.2,
    ):
        with h5py.File(data_path, "r") as f:
            self.images = f["image"]
            self.masks = f["mask"]
        
        self.train_dataset = StanfordDataset(
            
        )
        
        
    def train_loader(self, batch_size: int = 16):
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            
        )
    
    def val_loader(self, batch_size: int = 16):
        return DataLoader(
            
        )
    
class VOCSegLoader(DataLoader):
    def __init__(
        self,
        batch_size: int = 16,
        root: str = config.DATA_PATH,
        num_workers: int = config.NUM_WORKERS,
    ):
        train_transforms = None
        val_transforms = None
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_data = VOCSegmentation(
            root=root,
            image_set="train",
            download=True,
            transforms=train_transforms
            )
        
        self.val_data = VOCSegmentation(
            root=root,
            image_set="val",
            download=True,
            transforms=val_transforms
            )
        
    def trainloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=False
            )
        
    def valloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=False
            )