from torch.utils.data import Dataset
from typing import Tuple
import torch
import numpy as np
from typing import Optional, Callable

class StanfordDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        transforms: Callable,
        test_data: bool = False,
    ):
        assert images.shape[0] == masks.shape[0], "Number of images and number of masks must match"
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.test_data = test_data
        
    def __len__(
        self
    ):
        return self.images.shape[0]
        
    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        image = torch.tensor(self.images[index], dtype = torch.uint8)
        image = self.transforms(image)
        mask = self.images[mask]
        mask = np.where(mask == 0, 0, 1)
        return (image, mask) if self.test_data else image
        