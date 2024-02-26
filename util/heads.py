import torch.nn as nn
import torch
import numpy as np

class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = np.argmax(**params)
        elif name == "argmax2d":
            self.activation = np.argmax(dim=1, **params)
        elif name == "clamp":
            self.activation = torch.clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)
class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling = "avg", dropout = 0.2, activation = None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p = dropout, inplace = True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias = True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, activation=None, upsampling=4):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=4) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)
# class SegmentationHead(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
#         super().__init__()
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
#         self.activation = Activation(activation)
#     def forward(self, x):
#         x = self.conv2d(x)
#         x = self.upsampling(x)
#         # print('before activate:',x[0][0][50][50])
#         # print('before activate:',x[0][1][50][50])
#         x = self.activation(x)
#         # print('after activate:',x[0][0][50][50])
#         # print('after activate:',x[0][1][50][50])
#         return x
        
class InstanceSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3, activation=None, upsampling=1):
        super(InstanceSegmentationHead, self).__init__()
        # Convolution to predict class scores for each pixel
        self.class_conv = nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size, padding=kernel_size // 2)
        # Convolution to predict an embedding for each pixel
        self.embedding_conv = nn.Conv2d(in_channels, 256, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Identity()

    def forward(self, x):
        class_logits = self.class_conv(x)
        class_logits = self.upsampling(class_logits)
        
        embedding = self.embedding_conv(x)
        embedding = self.upsampling(embedding)
        
        if self.activation:
            class_logits = self.activation(class_logits)
        
        # Note: 這裡沒有實作像素嵌入後的聚類步驟，這通常是實例分割任務的一部分。
        # 在實際應用中，您需要新增後處理步驟，例如Mean Shift來根據嵌入將像素聚類到不同的實例中。
        
        return class_logits, embedding