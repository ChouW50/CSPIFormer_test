import torch, math
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from timm.models.layers import to_2tuple, DropPath
from einops.layers.torch import Rearrange, Reduce
import loralib as lora
class DWConv(nn.Module):
    def __init__(self, in_, out_, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_, out_, bias = True, groups = in_, **kwargs)
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace = True)
class PatchEmbed(nn.Module):
    def __init__(self, in_, out_, stride, patch_sizes):
        super().__init__()
        patch_size = to_2tuple(patch_sizes)
        self.proj = nn.Conv2d(in_, out_, kernel_size = patch_size, stride = stride, 
                                padding = (patch_size[0] // 2, patch_size[1] // 2))
        self.bn = nn.BatchNorm2d(out_)
    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        return x
class LoRA_MHSA(nn.Module):
    def __init__(self, dim, R, head = 8, qkv_bias = True, qk_scale = None, attn_drop_ratio = 0., linear_drop_ratio = 0.):
        super().__init__()
        
class Incep_MHSA(nn.Module):
    def __init__(self, dim, R, head = 8, qkv_bias = True, qk_scale = None, attn_drop_ratio = 0., linear_drop_ratio = 0.):
        super().__init__()
        self.head = head
        head_dim = dim // head
        self.scale = qk_scale or head_dim ** -0.5
        self.R = R
        self.q = nn.Linear(dim, dim, bias = qkv_bias)
        if R > 1:
            self.line1 = nn.Sequential(
                DWConv(dim, dim, kernel_size = (1, R), stride = (1, R)),
                DWConv(dim, dim, kernel_size = (R, 1), stride = (R, 1)),
            )
            self.line2 = DWConv(dim, dim, kernel_size = R, stride = R)
            self._33dwC = DWConv(dim, dim, kernel_size = 3, padding = 1)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias = qkv_bias)
        self.attn = nn.Softmax(dim = -1)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim, bias = qkv_bias)
        self.proj_drop = nn.Dropout(linear_drop_ratio)
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_layer = x.reshape(B, C, -1).permute(0, 2, 1)
        q = self.q(x_layer).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.R > 1:
            # line 1 set
            l1 = self.line1(x).view(B, C, -1)
            # line 2 set
            l2 = self.line2(x).view(B, C, -1)
            # line 3 set
            l3 = F.adaptive_avg_pool2d(x, (H // self.R, W // self.R))
            l3 = self._33dwC(l3).view(B, C, -1)
            
            l_ = torch.cat([l1, l2, l3], dim = 2)
            l_ = self.norm(l_.permute(0, 2, 1))
            
            kv = self.kv(l_).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_layer).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attns = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attns = self.attn(attns)
        attns = self.attn_drop(attns)
        out = (attns @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(out)
        x = self.proj_drop(out)
        return x
class EFFN(nn.Module):
    def __init__(self, in_, out_, hid_, drop = 0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_, hid_, kernel_size = 1)
        self.act = nn.GELU()
        self.dwc = DWConv(hid_, hid_, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(hid_, out_, kernel_size = 1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.dwc(x))
        x = self.act(self.conv2(x))
        x = self.drop(x)
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
class Block(nn.Module):
    def __init__(self, dim, out_dim, effn_ratio, R, head, dpr, drop_out = 0., add_LoRA = False):
        super().__init__()
        hid_dim = int(dim * effn_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.bn = nn.BatchNorm2d(dim)
        if add_LoRA:
            self.MHSA = LoRA_MHSA(dim, R, head = head, attn_drop_ratio = drop_out, linear_drop_ratio = drop_out)
        else:
            self.MHSA = Incep_MHSA(dim, R, head = head, attn_drop_ratio = drop_out, linear_drop_ratio = drop_out)
        self.EFFN_ = EFFN(dim, out_dim, hid_dim, drop_out)
    def forward(self, x):
        B, C, H, W = x.shape
        hid_ = self.drop_path(self.MHSA(self.bn(x))).permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        hid_ = x + hid_
        out_ = hid_ + self.drop_path(self.EFFN_(self.bn(hid_)))
        return out_
class Stage(nn.Module):
    def __init__(self, in_, out_, ratio, embed_dim, stride, patches, depth, R, head, dpr, index, add_LoRA = False):
        super().__init__()
        block = []
        self.patch = PatchEmbed(in_ = in_, out_ = embed_dim, stride = stride, patch_sizes = patches)
        for d in range(depth):
            block.append(Block(embed_dim, out_, ratio, R, head, dpr[index + d], add_LoRA = add_LoRA))
        self.block = nn.Sequential(*block)
    def forward(self, x):
        x = self.patch(x)
        x = self.block(x)
        return x
class SegHead(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.conv1 = nn.Conv2d(in_, out_, kernel_size = 1)
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv1(x)
        x = F.interpolate(x, size = (H*4, W*4), mode = 'bilinear', align_corners = False)
        return x
class ClassHead(nn.Module):
    def __init__(self, in_, hid_, num_class, size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_, hid_, kernel_size = 1)
        self.drop = nn.Dropout(0.1)
        self.flat = nn.Flatten()
        channel = math.prod(size) * hid_
        self.fc = nn.Linear(channel, num_class)
    def forward(self, x):
        x = self.drop(self.conv1(x))
        # x = x.mean(3).mean(2)
        x = self.flat(x)
        x = self.fc(x)
        return x
class TFEncoder(nn.Module):
    def __init__(self, in_ = 3, out_ = 512, depth = [2, 2, 4, 2], embed_dim = [64, 128, 320, 512], 
                    patch_size = [7, 3], effn_ratio = [8, 8, 4, 4], 
                    R = [8, 4, 2, 1], head = [2, 4, 8, 16], for_seg = True, add_RoLA = False):
        super().__init__()
        self.for_seg = for_seg
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depth))]  # stochastic depth decay rule
        self.stage1 = Stage(in_, embed_dim[0], effn_ratio[0], embed_dim[0], 4, patch_size[0], depth[0], R[0], head[0], dpr, 0, add_LoRA = add_RoLA)
        self.bn1 = nn.BatchNorm2d(embed_dim[0])
        self.stage2 = Stage(embed_dim[0], embed_dim[1], effn_ratio[1], embed_dim[1], 2, patch_size[1], depth[1], R[1], head[1], dpr, sum(depth[:1]), add_LoRA = add_RoLA)
        self.bn2 = nn.BatchNorm2d(embed_dim[1])
        self.stage3 = Stage(embed_dim[1], embed_dim[2], effn_ratio[2], embed_dim[2], 2, patch_size[1], depth[2], R[2], head[2], dpr, sum(depth[:2]), add_LoRA = add_RoLA)
        self.bn3 = nn.BatchNorm2d(embed_dim[2])
        self.stage4 = Stage(embed_dim[2], out_, effn_ratio[3], embed_dim[3], 2, patch_size[1], depth[3], R[3], head[3], dpr, sum(depth[:3]), add_LoRA = add_RoLA)
        self.bn4 = nn.BatchNorm2d(out_)
    def forward(self, x):
        s1 = self.stage1(x)
        s1 = self.bn1(s1)
        s2 = self.stage2(s1)
        s2 = self.bn2(s2)
        s3 = self.stage3(s2)
        s3 = self.bn3(s3)
        s4 = self.stage4(s3)
        s4 = self.bn4(s4)
        if self.for_seg:
            return s1, s2, s3, s4
        else:
            return s4
class TFDecoder(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.conv1 = nn.Conv2d(in_, out_, kernel_size = 1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.1)
    def do_ac(self, stage):
        ac = self.act(stage.mean(dim = 1).unsqueeze(dim = 1))
        feat = torch.mul(stage, ac)
        feat = stage + feat
        return feat
    def forward(self, s1, s2, s3, s4):
        s1 = self.do_ac(s1)
        # ↧ upsampling to s1 size 
        s2 = F.interpolate(self.do_ac(s2), size = s1.shape[2:], mode = 'bilinear', align_corners = False)
        s3 = F.interpolate(self.do_ac(s3), size = s1.shape[2:], mode = 'bilinear', align_corners = False)
        s4 = F.interpolate(s4, size = s1.shape[2:], mode = 'bilinear', align_corners = False)
        # ↥ end
        x = torch.cat([s1, s2, s3, s4], dim = 1)
        out_ = self.drop(self.conv1(x))
        return out_
class IncepFormer(nn.Module):
    def __init__(self, size = (512, 512), in_ = 3, hid_ = 512, out_ = 512, depth = [2, 2, 4, 2], num_class = 100, For_seg = True, RoLA_ = False, embed_dim = [64, 128, 320, 512]):
        super().__init__()
        self.for_seg = For_seg
        self.encoder = TFEncoder(in_ = in_, out_ = hid_, depth = depth, for_seg = For_seg, add_RoLA = RoLA_)
        self.decoder = TFDecoder(in_ = sum(embed_dim), out_ = out_)
        if For_seg:
            self.head = SegHead(out_, num_class)
        else:
            self.head = ClassHead(embed_dim[3], out_, num_class, (size[0] // 32, size[1] // 32))
    def forward(self, in_):
        if self.for_seg:
            s1, s2, s3, s4 = self.encoder(in_)
            out_ = self.decoder(s1, s2, s3, s4)
        else:
            out_ = self.encoder(in_)
        out_ = self.head(out_)
        return out_