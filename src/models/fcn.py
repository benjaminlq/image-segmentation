import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder: Callable,
        decoder: Callable,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, imgs):
        x = self.encoder(imgs)
        outs = self.decoder(x)
        return outs
    
class FCN(EncoderDecoder):
    def __init__(
        self,
        encoder: Callable,
        decoder: Callable,
    ):
        super(FCN, self).__init__(encoder, decoder)

class ViT