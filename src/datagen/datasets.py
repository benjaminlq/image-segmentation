from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
import torch
import numpy as np
from typing import Callable
from src.datagen.augment import MaskNormalize, MaskCompose, MaskHorizontalFlip

class StanfordDataset(Dataset):
    def __init__(
        self,
        images, 
        masks, 
        transforms: Callable,
        training: bool = True,
    ):
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.training = training
        
    def __len__(
        self
    ):
        return self.images.shape[0]
        
    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        image = torch.tensor(self.images[index], dtype = torch.uint8).unsqueeze(0)
        mask = torch.tensor(self.masks[index], dtype = torch.uint8).unsqueeze(0)
        image, mask = self.transforms(image, mask)
                
        return (image, mask.squeeze(0)) if self.training else image
        
if __name__ == "__main__":
    data_size = 200
    h, w = 802, 1054
    dummy_images = np.random.randint(0, 256, size = (data_size, h, w)).astype(np.uint8)
    dummy_masks = np.random.choice([0, 255], size = (data_size, h, w)).astype(np.uint8)
    test_transforms = MaskCompose(
        [
            MaskHorizontalFlip(),
            MaskNormalize(),
        ]
    )
    test_dataset = StanfordDataset(
        images=dummy_images, masks=dummy_masks, transforms=test_transforms
    )
    print(f"Test dataset length: {len(test_dataset)}")
    sample_img, sample_mask = test_dataset[0]
    print(type(sample_img))
    print(sample_img.size())
    print(sample_img.max())
    print(sample_img.min())
    print(type(sample_mask))
    print(sample_mask.size())
    print(sample_mask.max())
    print(sample_mask.min())