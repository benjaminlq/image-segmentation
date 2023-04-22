import torch
import torch.nn as nn
from torchvision import transforms
from blocks import DoubleConv, Down
from configs import DEVICE
from typing import Tuple