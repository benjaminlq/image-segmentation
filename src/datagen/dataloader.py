from torch.utils.data import DataLoader, SubsetRandomSampler
from src.datagen.augment import MaskCompose, MaskHorizontalFlip, MaskNormalize, MaskRandomResizeCrop, MaskRandomRotate
import os
import config
from src.datagen.datasets import StanfordDataset
import h5py
import random
import math
import numpy as np

class StanfordDataLoader(DataLoader):
    def __init__(
        self, 
        data_path: str = os.path.join(config.DATA_DIR, "dataset.hdf5"),
        augmentation: bool = True,
        val_ratio: float = 0.2,
    ):
        f = h5py.File(data_path, "r")
        self.images = f["image"]
        self.masks = f["mask"]
        # print("Finished Loading images to memory")
        self.total_size = self.images.shape[0]
        self.val_ratio = val_ratio
        self.val_transform = MaskCompose(
            [
                MaskNormalize(0.5, 0.5)
            ]
        )

        if augmentation:
            self.train_transform = MaskCompose(
                [
                    MaskHorizontalFlip(0.5),
                    MaskRandomResizeCrop(),
                    MaskRandomRotate(25),
                    MaskNormalize(0.5, 0.5)
                ]
            )
        else:
            self.train_transform = self.val_transform
        
        self.train_idx, self.val_idx = self.generate_split_idx()
        
        self.train_dataset = StanfordDataset(
            f["image"], f["mask"], transforms=self.train_transform
        )
        
        self.val_dataset = StanfordDataset(
            f["image"], f["mask"], transforms=self.val_transform
        )
        print("Dataloader Setup Completed")
    
    def generate_split_idx(self):
        idx = list(range(self.total_size))
        random.shuffle(idx)
        self.val_size = math.floor(self.total_size * self.val_ratio)
        self.train_size = self.total_size - self.val_size
        return idx[:self.train_size], idx[self.train_size:]
        
    def train_loader(self, batch_size = 16):
        return DataLoader(
            self.train_dataset, batch_size=batch_size,drop_last=True,num_workers=0,pin_memory=True, sampler=SubsetRandomSampler(self.train_idx)
        )
        
    def val_loader(self, batch_size = 16):
        return DataLoader(
            self.val_dataset, batch_size=batch_size,drop_last=False,num_workers=0,pin_memory=True, sampler=SubsetRandomSampler(self.val_idx)
        )
        
if __name__ == "__main__":
    datamanager = StanfordDataLoader()
    trainloader = datamanager.train_loader()
    valloader= datamanager.val_loader()
    imgs, masks = next(iter(trainloader))
    print(imgs.size())
    print(masks.size())