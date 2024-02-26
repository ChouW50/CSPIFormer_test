import torch.nn as nn
import sys, os
from models.CSP_IFormer_final_SegMode_v3 import iformer_small
# from models.CSPST_IFormer import iformer_small

from util.heads import ClassificationHead, SegmentationHead
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.BiFPN_2_CSPIformer import BiFPNDecoder

class CSPIformer2BiFPN(nn.Module):
    def __init__(self, size, num_classes, encoder_channels, win_sizes, split_sizes, ImageLabel = 0):
        super().__init__()
        self.ImageLabel = ImageLabel
        self.encoder = iformer_small(size = size, num_classes = 0, win_sizes = win_sizes, split_sizes = split_sizes)
        if ImageLabel == 1 or ImageLabel == 3:
            self.decoder = BiFPNDecoder(encoder_channels = encoder_channels)
            self.seghead = SegmentationHead(128, num_classes, activation='softmax2d')
        elif ImageLabel == 0 or ImageLabel == 2:
            self.seghead = ClassificationHead(encoder_channels[3], num_classes)
    def forward(self, features):
        x = self.encoder(features)
        if self.ImageLabel == 0 or self.ImageLabel == 2:
            x = x[3]
        elif self.ImageLabel == 1 or self.ImageLabel == 3:
            x = self.decoder(x)
            
        # print(f"x.shape: {x.shape}")
        x = self.seghead(x)
        return x