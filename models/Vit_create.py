import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # 將x的第一個維度保留，其他維度都變成1
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    # 生成一個大小為shape的隨機tensor，並加上keep_prob
    random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
    # 將隨機tensor取整數(一部分為0另一部分為1)
    random_tensor.floor_()
    # x除以keep_prob，再乘上隨機tensor，就可以得到一個隨機的x(一部分為0另一部分為原本的x)
    output = x.div(keep_prob) * random_tensor
    return output
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
def mlp_head(dim, num_class):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_class)
    )
class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_dim, patch_sizes):
        super().__init__()
        self.patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_sizes, p2 = patch_sizes)
        self.proj = nn.Linear(patch_dim, dim)
    def forward(self, x):
        x = self.patch(x)
        x = self.proj(x)
        return x
class MHSA(nn.Module):
    def __init__ (self, dim, head = 8, qkv_bias = False, qk_scale = None, attn_drop_ratio = 0., linear_drop_ratio = 0.):
        super().__init__()
        self.head = head
        self.head_dim = dim // head
        self.scale = qk_scale or dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        # softmax(dim = -1): 對最後一個維度做softmax
        self.attn = nn.Softmax(dim = -1)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(linear_drop_ratio)
    def forward(self, x):
        # B: batch size, N: number of patched, C: channels
        B, N, C = x.shape
        # qkv(x): [B, (197)N, (768)C] -> [B, 197, 768 * 3]
        # reshape: [B, (197)N, (768)C * (3)qkv] -> [B, 197, (3)qkv, (12)head, (64)head_dim]
        # permute: [B, (197)N, (3)qkv, (12)head, (64)head_dim] -> [(3)qkv, B, (12)head, 197, (64)head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attns = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attns = self.attn(attns)
        attns = self.attn_drop(attns)
        # attn @ v = [B, (12)head, 197, 197] @ [B, (12)head, 197, 64] -> [B, (12)head, 197, 64]
        # transpose(1, 2): [B, (12)head, 197, 64] -> [B, 197, (12)head, 64]
        # reshape: [B, 197, (12)head, 64] -> [B, 197, 768]
        out = (attns @ v).transpose(1, 2).reshape(B, N, C)
        # out = einsum('b h i j, b h j d -> b h i d', attns, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        x = self.proj(out)
        x = self.proj_drop(out)
        return x
class MLP(nn.Module):
    def __init__(self, in_, hid_, drop_ratio = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_, hid_)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_, in_)
        self.drop = nn.Dropout(drop_ratio)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class DropPath(nn.Module):
    def __init__(self, drop_prob = None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class TranformerEncoder(nn.Module):
    # depth: 幾層transformer
    def __init__(self, dim, depth, head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MHSA(dim, head = head, attn_drop_ratio = dropout, linear_drop_ratio = dropout)),
                PreNorm(dim, MLP(in_ = dim, hid_ = mlp_dim, drop_ratio = dropout))
                ]))
    def forward(self, x):
        # print(self.layers)
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        return x
class ViT(nn.Module):
    def __init__(self, img_size, num_class, dim, depth, patch_size = 16, mod_name = "ViT_Base", pool = 'cls', channels = 3, head_dim = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        img_h, img_w = pair(img_size)
        patch_h, patch_w = pair(patch_size)
        if mod_name == "ViT_Base":
            head = 12
            mlp_dim = 3072
        elif mod_name == "ViT_Large":
            head = 16
            mlp_dim = 4096
        elif mod_name == "ViT_Huge":
            head = 16
            mlp_dim = 5120
        assert img_h % patch_h == 0 and img_w % patch_w == 0, 'image dimensions must be divisible by the patch size'
        
        num_patch = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = channels * patch_h * patch_w
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_embed = PatchEmbed(dim, patch_dim, patch_size)
        self.PosEmbed = nn.Parameter(torch.randn(1, num_patch + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformerE = TranformerEncoder(dim, depth, head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.MLPHead = mlp_head(dim, num_class)
    def forward(self, img):
        x = self.patch_embed(img)
        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b = B)
        x = torch.cat([cls_token, x], dim = 1)
        x += self.PosEmbed[:, :(N + 1)]
        x = self.dropout(x)
        x = self.transformerE(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.MLPHead(x)
        return x