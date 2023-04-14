import torch
import torch.nn as nn
from torchvision import transforms
from blocks import DoubleConv, Down
from configs import DEVICE
from typing import Tuple

class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        n
    ):
        super(UNetDecoder, self).__init__()
        self.