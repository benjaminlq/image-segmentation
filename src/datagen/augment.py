import torch
from torchvision.transforms import RandomResizedCrop, ToTensor
import torchvision.transforms.functional as Fv
from typing import Union, Tuple, List
import numpy as np
import random

class MaskHorizontalFlip(object):
    def __init__(
        self,
        p: float = 0.5,
    ):
        self.p = p
        
    def __call__(
        self, image, mask
    ):
        if random.random() < self.p:
            transformed_img = Fv.hflip(image)
            transformed_mask = Fv.hflip(mask)
        else:
            transformed_img, transformed_mask = image, mask
        return transformed_img, transformed_mask
    
class MaskRandomResizeCrop(object):
    def __init__(
        self,
        size: Tuple[int, int] = (802, 1054),
        scale: Tuple[float, float] = (0.6, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        
    def __call__(
        self, image, mask
    ):
        i, j, h, w = RandomResizedCrop.get_params(image, self.scale, self.ratio)
        transformed_img = Fv.resized_crop(image, i, j, h, w, self.size, antialias=True)
        transformed_mask = Fv.resized_crop(mask, i, j, h, w, self.size, antialias=False)
        return transformed_img, transformed_mask
    
class MaskRandomRotate(object):
    def __init__(
        self,
        degree: Union[int, float, Tuple[float, float]] = 15
    ):
        if isinstance(degree, int) or isinstance(degree, float):
            self.degrees = (-degree, degree)
        else:
            self.degrees = degree
        self.fill = 0
        
    def __call__(
        self, image, mask
    ):
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]),
                                              float(self.degrees[1])).item())
        fill = self.fill
        if isinstance(image, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)]
            else:
                fill = [float(f) for f in fill]
        transformed_img = Fv.rotate(image, angle, fill = fill)
        transformed_mask = Fv.rotate(mask, angle, fill = fill)
        return transformed_img, transformed_mask
    
class MaskNormalize(object):
    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.mean = mean
        self.std = std
        
    def __call__(
        self, image, mask
    ):
        image = image / 255.0
        transformed_img = Fv.normalize(image, self.mean, self.std)
        transformed_mask = torch.where(mask==255.0, 1, 0)
        
        return transformed_img, transformed_mask
        
class MaskCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
        
if __name__ == "__main__":
    augment = MaskNormalize()
    sample_image = torch.randn(1, 200, 200)
    sample_mask = torch.randn(1, 200, 200)
    transformed_img, transformed_mask = augment(sample_image, sample_mask)
    print(transformed_img.size())
    print(transformed_mask.size())