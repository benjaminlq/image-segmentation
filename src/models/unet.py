import torch
import torch.nn as nn
from torchvision import transforms
from typing import Tuple
from blocks import DoubleConv, Down, Up
from configs import DEVICE

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        img_size: Tuple=(512, 512),
        n_classes: int=10,
    ):
        super(UNet, self).__init__()
        self.h, self.w = img_size[0], img_size[1]
        self.conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear=False)
        self.up2 = Up(512, 256, bilinear=False)
        self.up3 = Up(256, 128, bilinear=False)
        self.up4 = Up(128, 64, bilinear=False)
        
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.resize = transforms.Resize(size = (self.h, self.w))
        
    def forward(self, imgs):
        x = self.resize(imgs)
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up1(x5, x4)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up4(x2, x1)
        out = self.out(x1)
        return out

    def use_checkpointing(self):
        self.conv = torch.utils.checkpoint(self.conv)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.out = torch.utils.checkpoint(self.out)
        
if __name__ == "__main__":
    sample_imgs = torch.randn(size=(5,3,224,224), device = DEVICE)
    model = UNet()
    model.to(DEVICE)
    outs = model(sample_imgs)
    print(outs.size())