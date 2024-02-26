import torch, math
import torch.nn as nn
import torch.nn.functional as F

# from models.CSP_IFormer_final_SegMode import iformer_small
# from models.CSP_IFormer_finel_SegMode_v3 import iformer_small
# from util.heads import ClassificationHead, SegmentationHead

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2,
                                mode="bilinear", align_corners=True)
        return x


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 


    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                    padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels,
                                upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(
                    out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    self.policy)
            )


class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        # self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)


        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)        

        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)

        # TODO: Init weights
        # self.w1 = nn.Parameter(torch.Tensor(2, 3))
        # self.w1_relu = nn.ReLU()
        # self.w2 = nn.Parameter(torch.Tensor(3, 3))
        # self.w2_relu = nn.ReLU()

    def forward(self, inputs):
        p4_x, p5_x, p6_x, p7_x = inputs
        
        #td
        p7_td = p7_x
        p6_td = self.p6_td(self.p6_td_w1 * p6_x + self.p6_td_w2 *F.interpolate(p7_x, scale_factor=2))
        p5_td = self.p5_td(self.p5_td_w1 * p5_x + self.p5_td_w2 *F.interpolate(p6_x, scale_factor=2))
        p4_td = self.p4_td(self.p4_td_w1 * p4_x + self.p4_td_w2 *F.interpolate(p5_x, scale_factor=2))

        #out
        p4_out = p4_td
        p5_out = self.p5_out(self.p5_out_w1 * p5_x + self.p5_out_w2 * p5_td + self.p5_out_w3 * F.interpolate(p4_out, scale_factor=0.5))
        p6_out = self.p6_out(self.p6_out_w1 * p6_x + self.p6_out_w2 * p6_td + self.p6_out_w3 * F.interpolate(p5_out, scale_factor=0.5))
        p7_out = self.p7_out(self.p7_out_w1 * p7_x + self.p7_out_w2 * p7_td + self.p7_out_w3 * F.interpolate(p6_out, scale_factor=0.5))

        return [p4_out, p5_out, p6_out, p7_out]


class BiFPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=4,
            feature_size=64,
            num_layers=3,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
            epsilon=0.0001
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]
        size = encoder_channels
        # self.p3 = nn.Conv2d(size[2], feature_size,
        #                     kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[3], feature_size,
                            kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[2], feature_size,
                            kernel_size=1, stride=1, padding=0)

        self.p6 = nn.Conv2d(size[1], feature_size,
                            kernel_size=1, stride=1, padding=0)

        self.p7 = ConvBlock(size[0], feature_size,
                            kernel_size=1, stride=1, padding=0)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(
                feature_size, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [0, 1, 2, 3]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, features):
        # c2, c3, c4, c5 = features[::-1]
        c2, c3, c4, c5 = features
        # p3_x = self.p3(c3)
        p4_x = self.p4(c2)
        p5_x = self.p5(c3)
        p6_x = self.p6(c4)
        p7_x = self.p7(c5)
        features = [p4_x, p5_x, p6_x, p7_x]
        [p4_out, p5_out, p6_out, p7_out] = self.bifpn(features)
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p4_out, p5_out, p6_out, p7_out])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        # _, _, H, W = x.shape
        # x = F.interpolate(x, size = (H*4, W*4), mode = "bilinear", align_corners = False)  
        # print(f"x.shape: {x.shape}")
        return x


# class CSPIformer2BiFPN(nn.Module):
#     def __init__(self, size, num_classes, encoder_channels, win_sizes, split_sizes, ImageLabel = False):
#         super().__init__()
#         self.ImageLabel = ImageLabel
#         self.encoder = iformer_small(size = size, num_classes = 0, win_sizes = win_sizes, split_sizes = split_sizes)
#         # self.decoder = BiFPNDecoder(encoder_channels = encoder_channels)
#         if ImageLabel:
#             self.seghead = SegmentationHead(128, num_classes)
#         else:
#             self.seghead = ClassificationHead(encoder_channels[3], num_classes)
#     def forward(self, features):
#         x = self.encoder(features)
#         if self.ImageLabel == False:
#             x = x[3]
#         # print(f"x.shape: {x.shape}")
#         # x = self.decoder(x)
#         x = self.seghead(x)
#         return x
