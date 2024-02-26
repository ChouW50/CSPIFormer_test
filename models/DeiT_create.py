import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class patchEmbedding(nn.Module):
    def __init__(self, in_channel = 3, patch_size = 16, embed_dim = 768, img_size_w = 224, img_size_h = 224):
        super().__init__()
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dist_tooken = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position = nn.Parameter(torch.randn((img_size_w // patch_size) * (img_size_h // patch_size) + 1, embed_dim))
    def forward(self, x):
        