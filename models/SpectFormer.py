import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import math
import numpy as np

class spectral_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        