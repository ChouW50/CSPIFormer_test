import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

from timm.models.layers import to_2tuple

'''
with 
Channel Shuffle
DropKey
CS+DK
'''


def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
        as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
def img2win(img, H_, W_):
    """
    img: B, C, H, W
    """
    B, C, H, W = img.shape
    # img_reshape(-1, H_ * W_, C) -> B, H_ * W_, C
    img_reshape = img.view(B, C, H // H_, H_, W // W_, W_)
    # B, H' * W', C
    img_reshape = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_ * W_, C)
    return img_reshape

def win2img(img_splits_HW, H_, W_, H, W):
    """
    img_splits_HW: B', H, W, C
    """
    B = int(img_splits_HW.shape[0] / (H * W / H_ / W_)) 
    img = img_splits_HW.view(B, H // H_, W //W_, H_, W_, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, kernel_size=16,  stride=16, padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x


# IFormer first patch embedding
class IFormer_FirstPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding"""

    def __init__(self, kernel_size=3,  stride=2, padding=1, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj1 = nn.Conv2d(
            in_chans, embed_dim//2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(embed_dim // 2)
        self.gelu1 = nn.GELU()
        self.proj2 = nn.Conv2d(
            embed_dim//2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm2 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.proj2(x)
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1)
        return x

# CMT first patch embedding


class HighMixer(nn.Module):
    """
    linear
    maxpool
    high frequency feature extractor
    """

    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()

        self.cnn_in = cnn_in = dim // 2
        self.pool_in = pool_in = dim // 2

        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2
        # dwconv
        self.conv1 = nn.Conv2d(
            cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False, groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()
        #maxpool
        self.Maxpool = nn.MaxPool2d(
            kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(
            pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # B, C, H, W
        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)
        
        hx = torch.cat((cx, px), dim=1)
        return hx


class LowMixer(nn.Module):
    """
    Transformer
    low frequency feature extractor
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0,
                                    count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(
            scale_factor=pool_size) if pool_size > 1 else nn.Identity()

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # ----dropkey start---- (make the classification work be better with catdog dataset)
        # m_r = torch.ones_like(attn)*0.1
        # attn = attn + torch.bernoulli(m_r)*-1e12
        # -----dropkey end-----
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x
    
    def forward(self, x):
        # B, C, H, W
        B, _, _, _ = x.shape
        xa = self.pool(x)
        xa = xa.permute(0, 2, 3, 1)
        B, H, W, C = xa.shape
        N = H * W
        # ↧ original
        # xa = xa.permute(0, 2, 3, 1).view(B, -1, self.dim)
        # B, N, C = xa.shape
        # ↥
        qkv = self.qkv(xa)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N**0.5), int(N**0.5))  # .permute(0, 3, 1, 2)

        xa = self.uppool(xa)

        return xa

# class SWinAttention(nn.Module):
#     def __init__(self, dim, dim_out, win_size, qk_scale, attn_drop, num_heads, res):
#         super().__init__()
#         self.dim = dim
#         self.dim_out = dim_out
#         self.num_heads = num_heads
#         self.scale = qk_scale
#         self.win_size = win_size
#         self.res = res
#         self.attn_drop = attn_drop
#         self.re_pos = nn.Parameter(
#             torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))
#         w_H = torch.arange(win_size[0])
#         w_W = torch.arange(win_size[1])
#         win = torch.stack(torch.meshgrid([w_H, w_W])) # 2, H_, W_
#         win_flatten = torch.flatten(win, 1) # 2, H_ * W_
        
#         # re_w = win_flatten.unsqueeze(2) - win_flatten.unsqueeze(1)
#         # re_w = re_w.transpose([1, 2, 0])
#         re_w = win_flatten[:, :, None] - win_flatten[:, None, :]  # 2, H_ * W_, H_ * W_
#         re_w = re_w.permute(1, 2, 0).contiguous()  # H_ * W_, H_ * W_, 2
#         re_w[:, :, 0] += win_size[0] - 1
#         re_w[:, :, 1] += win_size[1] - 1
#         re_w[:, :, 0] *= 2 * win_size[1] - 1
#         self.re_w_index = re_w.sum(-1) # H_ * W_, H_ * W_
#     def ori2winsize(self, qkv):
#         split_1 = (self.res - self.win_size[0]) // 2
#         split_2 = self.res - split_1 - self.win_size[0]
#         print(f"ori2winsize: {qkv.shape}, {qkv}")
#         R_, cen, L_ = qkv.split([split_1, self.win_size[0], split_2], 1)
#         U_, qkv, D_ = cen.split([split_1, self.win_size[0], split_2], 2)
#         return qkv, R_, L_, U_, D_
#     def winsize2ori(self, qkv, R_, L_, U_, D_):
#         cen = torch.cat([R_, qkv, L_], dim = 0)
#         qkv = torch.cat([U_, cen, D_], dim = 1)
#         return qkv
#     def forward(self, qkv):
#         print(f"SWin qkv: {qkv.shape}")
#         q, k, v = qkv.unbind(0)
#         print(f"q: {q.shape}")
#         B, L, C = q.shape
#         q, R_, L_, U_, D_ = self.ori2winsize(q)
#         print(f"q: {q.shape}, R_: {R_.shape}, L_: {L_.shape}, U_: {U_.shape}, D_: {D_.shape}")
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         print(f"attn: {attn.shape}")
#         re_pos_b = self.re_pos[self.re_w_index.view(-1)].view(self.win_size[0] * self.win_size[1], 
#                                                               self.win_size[0] * self.win_size[1], -1)
#         print(f"2. {re_pos_b.shape}")
#         re_pos_b = re_pos_b.permute(2, 0, 1).contiguous()
#         print(f"3. {re_pos_b.shape}")
#         print(re_pos_b.unsqueeze(0).shape)
#         attn = attn + re_pos_b.unsqueeze(0)
#         print(f"4. {attn.shape}")
#         attn = self.attn_drop(attn)
#         print(f"5. {attn.shape}")
#         x = (attn @ v).transpose(1, 2).reshape(B, L, C)
#         print(f"6. {x.shape}")
#         return x

class CSWinAttention(nn.Module):
    def __init__(self, dim, res, idx, split_size, num_heads, dim_out, qk_scale, attn_drop):
        super().__init__()
        self.dim = dim
        self.res = res
        self.dim_out = dim_out
        self.split_size = split_size
        self.num_heads = num_heads
        self.scale = qk_scale
        self.attn_drop = attn_drop
        if idx == 0:
            H_, W_ = self.res, self.split_size
        elif idx == 1:
            H_, W_ = self.split_size, self.res
        else:
            print(f"idx error→{idx}")
            exit(0)
        self.H_, self.W_ = H_, W_
        self.get_v = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)
    def i2w(self, img):
        B, N, C = img.shape
        H = W = int(np.sqrt(N))
        img = img.transpose(-2, -1).contiguous().view(B, C, H, W)
        img = img2win(img, self.H_, self.W_) # B, H' * W', C
        img = img.reshape(-1, self.H_ * self.W_, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return img
    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = x.view(B, C, H // self.H_, self.H_, W // self.W_, self.W_).permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, self.H_, self.W_) # B', C, H', W'
        lepe = func(x) # B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, self.H_ * self.W_).permute(0, 1, 3, 2).contiguous() # B', num_heads, H' * W', C // num_heads
        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_ * self.W_).permute(0, 1, 3, 2).contiguous() # B', num_heads, H' * W', C // num_heads
        return x, lepe
    def forward(self, qkv):
        q, k, v = qkv.unbind(0)
        H = W = self.res
        B, L, C = q.shape
        assert L == H * W, "input feature has wrong size"
        q = self.i2w(q)
        k = self.i2w(k)
        v, lepe = self.get_lepe(v, self.get_v)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        print(f"(CSWin)attn: {attn.shape}")
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_ * self.W_, C) # B', H' * W', C
        x = win2img(x, self.H_, self.W_, H, W) #B, H', W', C
        # x = win2img(x, self.H_, self.W_, H, W).view(B, -1, C) # Original
        return x

class AxwinLowMixear(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2, win_sizes = (2, 2), res = None, 
                    split_sizes = 7, **kwargs):
        super().__init__()
        # num_head: [1]*3, [3]*3, [7]*4...
        if num_heads == 1:
            self.h1, self.h2 = 2, 3
        elif num_heads == 3:
            self.h1, self.h2 = 4, 6
        # self.num_heads = num_heads // 2
        # SWin head_dim for scale
        self.head_dim = head_dim = dim // self.h2
        self.res = res
        self.split_sizes = split_sizes
        self.scale = head_dim ** -0.5 # qk_scale
        self.dim = dim
        self.win_sizes = win_sizes
        self.Twolines_dim = Twolines_dim = self.dim // 2
        self.proj1 = nn.Conv2d(self.dim, Twolines_dim, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.qkv_up = nn.Linear(Twolines_dim, Twolines_dim * 3, bias = qkv_bias)
        self.proj2 = nn.Conv2d(self.dim, Twolines_dim, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.qkv_dn = nn.Linear(Twolines_dim, Twolines_dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.attns_CSWin = nn.ModuleList([
            CSWinAttention(
                self.Twolines_dim // 2, self.res, idx = i, split_size = self.split_sizes, num_heads = self.h1 // 2, 
                dim_out = self.Twolines_dim // 2, qk_scale = (dim // self.h1 // 2) ** -0.5, attn_drop = self.attn_drop,
            )for i in range(2)
        ])
        # self.attns_SWin = nn.ModuleList([
        #     SWinAttention(
        #         self.Twolines_dim, self.Twolines_dim, self.win_sizes, self.scale, self.attn_drop, 
        #         self.h2, self.res
        #     )
        # ])
        
    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x
    
    def forward(self, xa):
        # B, C, H, W
        H = W = self.res
        print(f"res: {self.res}, dim: {self.dim}, Twolines_dim: {self.Twolines_dim}")
        xa_up = self.proj1(xa)
        xa_dn = self.proj2(xa)
        xa_up = xa_up.permute(0, 2, 3, 1)
        xa_dn = xa_dn.permute(0, 2, 3, 1)
        B, C, H_, W_ = xa.shape
        print(f"xa: {xa.shape}, xa_up: {xa_up.shape}")
        _, _, _, C_ = xa_up.shape
        assert H_ * W_ == H * W, "input feature has wrong size"
        qkv_1 = self.qkv_up(xa_up)
        qkv_1 = qkv_1.reshape(B, -1, 3, C_).permute(2, 0, 1, 3)
        print(f"qkv_1: {qkv_1.shape}")
        x1 = self.attns_CSWin[0](qkv_1[:, :, :, : C // 4]) # row
        x2 = self.attns_CSWin[1](qkv_1[:, :, :, C // 4: C // 2]) # column
        line_up = torch.cat([x1, x2], dim = 2)
        print(f"finish CSWin{x1.shape}, {x2.shape}, {line_up.shape}")
        qkv_2 = self.qkv_dn(xa_dn).reshape(B, -1, 3, C_).permute(2, 0, 1, 3)
        # qkv_2 = self.qkv_dn(xa_dn).reshape(B, H_ * W_, 3, self.h2, C_ // self.h2).permute(2, 0, 3, 1, 4)
        print(f"qkv_2: {qkv_2.shape}")
        line_dn = self.attns_SWin[0](qkv_2) # SWin
        print(f"finish SWin{line_dn.shape}")
        att_x = torch.cat([line_up, line_dn], dim = 2)
        print(f"finish SWin{att_x.shape}")
        lower = att_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        # qkv = self.qkv(xa)
        # qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # # make torchscript happy (cannot use tensor as tuple)
        # q, k, v = qkv.unbind(0)
        # xa = self.att_fun(q, k, v, B, N, C)
        # xa = xa.view(B, C, int(N**0.5), int(N**0.5))  # .permute(0, 3, 1, 2)

        return lower
class Inception_token_mixer(nn.Module):
    """
    Inception token mixer
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2, win_sizes = (2, 2),
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads

        self.low_dim = low_dim = attention_head * head_dim
        self.high_dim = high_dim = dim - low_dim

        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, num_heads=attention_head,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size, win_sizes = win_sizes)
        
        self.conv_fuse = nn.Conv2d(low_dim+high_dim*2, low_dim+high_dim*2, kernel_size=3,
                                    stride=1, padding=1, bias=False, groups=low_dim+high_dim*2)
        self.proj = nn.Conv2d(low_dim+high_dim*2, dim,
                                kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)

        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.low_mixer(lx)

        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class CSP_Inception_token_mixer(nn.Module):
    """
    Inception token mixer
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, 
                    pool_size=2, win_sizes = None, split_sizes = None, res = None,
                    **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads // 2

        self.low_dim = low_dim = attention_head * head_dim
        self.high_dim = high_dim = dim - low_dim

        self.high_mixer = HighMixer(high_dim)
        if split_sizes is None:
            self.low_mixer = LowMixer(low_dim, num_heads=attention_head,
                                        qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size)
        else:
            self.low_mixer = AxwinLowMixear(low_dim, num_heads=attention_head,
                                        qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size, win_sizes = win_sizes, 
                                        split_sizes = split_sizes, res = res)
        self.conv_fuse = nn.Conv2d(low_dim+high_dim*2, low_dim+high_dim*2, kernel_size=3,
                                    stride=1, padding=1, bias=False, groups=low_dim+high_dim*2)
        self.proj = nn.Conv2d(low_dim+high_dim*2, dim,
                                kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)


        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)


        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.low_mixer(lx)


        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class IFormer_Block(nn.Module):
    """
    IFormer block
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_head=1, pool_size=2,
                    attn=Inception_token_mixer,
                    use_layer_scale=False, layer_scale_init_value=1e-5, win_sizes = None, split_sizes = None, res = None,
                    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, attention_head=attention_head, 
                            pool_size=pool_size, win_sizes = win_sizes, split_sizes = split_sizes, res = res)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop)

        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            # print('use layer scale init value {}'.format(layer_scale_init_value))
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 *
                                    self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2 *
                                    self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CSP_IFomerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr, norm_layer, act_layer, 
                    attention_heads, pool_size, use_layer_scale, layer_scale_init_value, idx1, idx2, win_sizes = None, 
                    split_sizes = None, part_ratio=0.5, res = None, **kwargs):
        super(CSP_IFomerBlock, self).__init__()
        self.part1_chnls = int(dim*part_ratio)
        self.part2_chnls = dim-self.part1_chnls
        self.first_num_heads = int(num_heads*part_ratio)
        self.dense = nn.Sequential(*[
            IFormer_Block(
                dim=self.part1_chnls, num_heads=self.first_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[
                    i], norm_layer=norm_layer, act_layer=act_layer, attention_head=attention_heads[i], pool_size=pool_size,
                attn=CSP_Inception_token_mixer,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                win_sizes = win_sizes, split_sizes = split_sizes, res = res,
            )
            for i in range(idx1, idx2)])
    def forward(self, x):

        part1 = x[:, :, :, :self.part1_chnls]
        part2 = x[:, :, :, self.part1_chnls:]
        
        part2 = self.dense(part2)
        x = torch.cat([part1, part2], dim=3)
        x = x.permute(0, 3, 1, 2)
        # ---channel shuffle start---
        x = shuffle_chnls(x, 8)
        # ----channel shuffle end----
        return x


class InceptionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=None, depths=None,
                    num_heads=None, mlp_ratio=4., qkv_bias=True,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                    act_layer=None, weight_init='', 
                    attention_heads=None,
                    use_layer_scale=False, layer_scale_init_value=1e-5,
                    checkpoint_path=None, win_sizes = None, split_sizes = None, **kwargs
                    ):
        super().__init__()
        st2_idx = sum(depths[:1])
        st3_idx = sum(depths[:2])
        st4_idx = sum(depths[:3])
        depth = sum(depths)

        self.num_classes = num_classes

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.patch_embed = IFormer_FirstPatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0])
        self.num_patches1 = num_patches = img_size // 4
        self.pos_embed1 = nn.Parameter(torch.zeros(
            1, num_patches, num_patches, embed_dims[0]))
        self.blocks1 = CSP_IFomerBlock(
            dim=embed_dims[0],
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dpr=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            attention_heads=attention_heads,
            pool_size=2,
            use_layer_scale=False,
            layer_scale_init_value=1e-5,
            idx1=0,
            idx2=st2_idx,
            win_sizes = win_sizes, split_sizes = split_sizes,
            res =  img_size // 4, 
        )

        self.patch_embed2 = embed_layer(
            kernel_size=3, stride=2, padding=1, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.num_patches2 = num_patches = num_patches // 2
        self.pos_embed2 = nn.Parameter(torch.zeros(
            1, num_patches, num_patches, embed_dims[1]))
        self.blocks2 = CSP_IFomerBlock(
            dim=embed_dims[1],
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dpr=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            attention_heads=attention_heads,
            pool_size=2,
            use_layer_scale=False,
            layer_scale_init_value=1e-5,
            idx1=st2_idx,
            idx2=st3_idx,
            win_sizes = win_sizes, split_sizes = split_sizes,
            res =  img_size // 8, 
        )

        self.patch_embed3 = embed_layer(
            kernel_size=3, stride=2, padding=1, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.num_patches3 = num_patches = num_patches // 2
        self.pos_embed3 = nn.Parameter(torch.zeros(
            1, num_patches, num_patches, embed_dims[2]))
        self.blocks3 = CSP_IFomerBlock(
            dim=embed_dims[2],
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dpr=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            attention_heads=attention_heads,
            pool_size=1,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            idx1=st3_idx,
            idx2=st4_idx,
        )

        self.patch_embed4 = embed_layer(
            kernel_size=3, stride=2, padding=1, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.num_patches4 = num_patches = num_patches // 2
        self.pos_embed4 = nn.Parameter(torch.zeros(
            1, num_patches, num_patches, embed_dims[3]))
        self.blocks4 = CSP_IFomerBlock(
            dim=embed_dims[3],
            num_heads=num_heads[3],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dpr=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            attention_heads=attention_heads,
            pool_size=1,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            idx1=st4_idx,
            idx2=depth,
        )

        self.norm = norm_layer(embed_dims[-1])
        # Classifier head(s)
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        # set post block, for example, class attention layers

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)

        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(
                self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, num_patches_def, H, W):
        if H * W == num_patches_def * num_patches_def:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").permute(0, 2, 3, 1)

    def forward_features(self, x):
        output = []
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W)
        x = self.blocks1(x)

        # x = x.permute(0, 3, 1, 2)
        output.append(x)
        x = self.patch_embed2(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W)
        x = self.blocks2(x)

        # x = x.permute(0, 3, 1, 2)
        output.append(x)
        x = self.patch_embed3(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W)
        x = self.blocks3(x)

        # x = x.permute(0, 3, 1, 2)
        output.append(x)
        x = self.patch_embed4(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W)
        x = self.blocks4(x)
        x = x.permute(0, 3, 2, 1)

        # x = x.flatten(1, 2)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        output.append(x)
    
        return output

    def forward(self, x):
        x = self.forward_features(x)
        # x1 = self.head(x)
        return x


def iformer_small(pretrained=False, size=256, in_channel=3, **kwargs):
    """
    19.866M  4.849G 83.382
    """
    depths = [3, 3, 9, 3]
    embed_dims = [96, 192, 320, 384]
    num_heads = [3, 6, 10, 12]
    attention_heads = [1]*3 + [3]*3 + [7] * 4 + [9] * 5 + [11] * 3
    
    model = InceptionTransformer(img_size=size,
                                    in_chans=in_channel,
                                    depths=depths,
                                    embed_dims=embed_dims,
                                    num_heads=num_heads,
                                    attention_heads=attention_heads,
                                    use_layer_scale=True, layer_scale_init_value=1e-6, **kwargs)
    return model