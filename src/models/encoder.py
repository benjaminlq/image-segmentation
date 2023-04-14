import torch
import torch.nn as nn
from torchvision import transforms
from blocks import DoubleConv, Down
from configs import DEVICE
from typing import Tuple

class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int=1,
        img_size: Tuple=(512, 512),
        n_layers: int=4,
    ):
        super(UNetEncoder, self).__init__()
        self.h, self.w = img_size[0], img_size[1]
        layers = []
        out_channels = 64
        layers.append(DoubleConv(in_channels, out_channels))
        for _ in range(n_layers):
            in_channels = out_channels
            out_channels = out_channels * 2
            layers.append(Down(in_channels, out_channels))
        
        self.encoder = nn.Sequential(*layers)
        self.resize = transforms.Resize(size=(self.h, self.w))
        
    def forward(self, imgs):
        # bs, c, h, w
        x = self.resize(imgs)
        out = self.encoder(x)
        return out
    
if __name__ == "__main__":
    print(f"Mode device is {DEVICE}")
    sample_imgs = torch.randn(size=(5, 3, 224, 224), device = DEVICE)
    model = UNetEncoder(in_channels=3)
    model.to(DEVICE)
    sample_outs = model(sample_imgs)
    print(sample_outs.size())