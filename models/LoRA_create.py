import math, torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.CSP_IFormer_final_SegMode import iformer_small

class LoRA_qkv(nn.Module):
    def __init__(self, qkv, l_Aq, l_Bq, l_Ak, l_Bk, l_Av, l_Bv, add_):
        super().__init__()
        self.qkv = qkv
        self.l_Aq = l_Aq
        self.l_Bq = l_Bq
        self.l_Ak = l_Ak
        self.l_Bk = l_Bk
        self.l_Av = l_Av
        self.l_Bv = l_Bv
        self.dim = qkv.in_features
        self.add_ = add_
        # self.w = torch.eye(qkv.in_features)
    def forward(self, x):
        qkv = self.qkv(x)
        if 'q' in self.add_:
            new_q = self.l_Bq(self.l_Aq(x))
            qkv[:, :, :, : self.dim] += new_q
        if 'k' in self.add_:
            new_k = self.l_Bk(self.l_Ak(x))
            qkv[:, :, :, self.dim:-self.dim] += new_k
        if 'v' in self.add_:
            new_v = self.l_Bv(self.l_Av(x))
            qkv[:, :, :, -self.dim:] += new_v
        return qkv
class LoRA_o(nn.Module):
    def __init__(self, proj, l_Ao, l_Bo):
        super().__init__()
        self.proj = proj
        self.l_Ao = l_Ao
        self.l_Bo = l_Bo
    def forward(self, x):
        proj = self.proj(x)
        new_o = self.l_Bo(self.l_Ao(x))
        proj += new_o
        return proj

class LoRA_MHSA(nn.Module):
    def __init__(self, in_, r = 4, stage = 1, add_ = ['q', 'v'], requ_grad = False):
        super().__init__()
        assert r > 0
        # Only for CSPIformer
        if stage == 4:
            self.lora_layer = [len(in_.encoder.blocks1.dense), len(in_.encoder.blocks2.dense), len(in_.encoder.blocks3.dense), len(in_.encoder.blocks4.dense)]
            self.block_list = [in_.encoder.blocks1.dense, in_.encoder.blocks2.dense, in_.encoder.blocks3.dense, in_.encoder.blocks4.dense]
        for param in in_.encoder.parameters():
            param.requires_grad = requ_grad
        for param in in_.decoder.parameters():
            param.requires_grad = requ_grad
        self.r = r
        self.add_ = add_
        # save linear list (create for storage, then we can init them or load weights)
        self.w_As, self.w_Bs = [], []
        for index in range(stage):
            for layer_, block in enumerate(self.block_list[index]):
                if layer_ >= self.lora_layer[index]:
                    continue
                linear_Wqkv = block.attn.low_mixer.qkv
                object_conv = block.attn.proj
                linear_mlp1 = block.mlp.fc1
                linear_mlp2 = block.mlp.fc2
                # print(f"before: {linear_Wqkv}")
                self.dim = linear_Wqkv.in_features
                self.c_in = object_conv.in_channels
                self.c_out = object_conv.out_channels
                self.mlp_in = linear_mlp1.in_features
                self.mlp_out = linear_mlp1.out_features
                linear_WAq = nn.Linear(self.dim, self.r, bias = False)
                linear_WBq = nn.Linear(self.r, self.dim, bias = False)
                linear_WAk = nn.Linear(self.dim, self.r, bias = False)
                linear_WBk = nn.Linear(self.r, self.dim, bias = False)
                linear_WAv = nn.Linear(self.dim, self.r, bias = False)
                linear_WBv = nn.Linear(self.r, self.dim, bias = False)
                linear_WAo = nn.Conv2d(self.c_in, self.r, 1, bias = False)
                linear_WBo = nn.Conv2d(self.r, self.c_out, 1, bias = False)
                linear_WAmlp1 = nn.Linear(self.mlp_in, self.r, bias = False)
                linear_WBmlp1 = nn.Linear(self.r, self.mlp_out, bias = False)
                linear_WAmlp2 = nn.Linear(self.mlp_out, self.r, bias = False)
                linear_WBmlp2 = nn.Linear(self.r, self.mlp_in, bias = False)
                if 'q' in self.add_:
                    self.w_As.append(linear_WAq)
                    self.w_Bs.append(linear_WBq)
                    
                if 'k' in self.add_:
                    self.w_As.append(linear_WAk)
                    self.w_Bs.append(linear_WBk)
                    
                if 'v' in self.add_:
                    self.w_As.append(linear_WAv)
                    self.w_Bs.append(linear_WBv)
                
                if 'o' in self.add_:
                    # assert ('q' and 'k' and 'v' in self.add_), 'You should also add q, k, v to use o'
                    self.w_As.append(linear_WAo)
                    self.w_Bs.append(linear_WBo)
                    block.attn.proj = LoRA_o(object_conv, linear_WAo, linear_WBo)
                
                if 'mlp' in self.add_:
                    self.w_As.append(linear_WAmlp1)
                    self.w_As.append(linear_WAmlp2)
                    self.w_Bs.append(linear_WBmlp1)
                    self.w_Bs.append(linear_WBmlp2)
                    block.mlp.fc1 = LoRA_o(linear_mlp1, linear_WAmlp1, linear_WBmlp1)
                    block.mlp.fc2 = LoRA_o(linear_mlp2, linear_WAmlp2, linear_WBmlp2)
                if 'q' or 'k' or 'v' in self.add_:
                    block.attn.low_mixer.qkv = LoRA_qkv(linear_Wqkv, linear_WAq, 
                                                    linear_WBq, linear_WAk, 
                                                    linear_WBk, linear_WAv, 
                                                    linear_WBv, self.add_)
        self.reset_parameters()
        self.FT = in_
        
    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.FT, torch.nn.DataParallel) or isinstance(self.FT, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.FT.module.state_dict()
        else:
            state_dict = self.FT.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = nn.parameter.Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = nn.parameter.Parameter(saved_tensor)

        sam_dict = self.FT.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.FT.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a = math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
    def forward(self, x):
        return self.FT(x)
def add_lora(model = iformer_small(), r = 4, stage = 4):
    model = model
    lora_model = LoRA_MHSA(model, r, stage)
    return lora_model
