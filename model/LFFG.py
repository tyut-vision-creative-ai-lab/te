import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# -----------------------------
# Haar DWT / IDWT   小波变换加SA
# -----------------------------

def dwt2_haar(x: torch.Tensor):
    """
    x: [B, C, H, W]  (H, W must be even)
    return: LL, HL, LH, HH each [B, C, H//2, W//2]
    """
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for Haar DWT"

    x01 = x[:, :, 0::2, :] * 0.5
    x02 = x[:, :, 1::2, :] * 0.5
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return LL, HL, LH, HH


def idwt2_haar(LL: torch.Tensor, HL: torch.Tensor, LH: torch.Tensor, HH: torch.Tensor):
    """
    inverse of dwt2_haar
    inputs: [B, C, h, w] -> output: [B, C, 2h, 2w]
    """
    a = (LL - HL - LH + HH) * 0.5
    b = (LL - HL + LH - HH) * 0.5
    c = (LL + HL - LH - HH) * 0.5
    d = (LL + HL + LH + HH) * 0.5

    B, C, h, w = LL.shape
    x = torch.zeros(B, C, h * 2, w * 2, device=LL.device, dtype=LL.dtype)
    x[:, :, 0::2, 0::2] = a
    x[:, :, 1::2, 0::2] = b
    x[:, :, 0::2, 1::2] = c
    x[:, :, 1::2, 1::2] = d
    return x


class LayerNorm2d(nn.Module):
    """
    LayerNorm wrapper for NCHW tensors.

    Applies LayerNorm over the channel dimension for each spatial location:
      input:  [B, C, H, W]
      internals: permute -> [B, H, W, C], apply nn.LayerNorm(C), permute back

    This is commonly used instead of BatchNorm2d when batch size is small.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> LN over C for each (b,h,w)
        # permute to [B, H, W, C]
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_perm)
        # permute back to [B, C, H, W]
        return x_norm.permute(0, 3, 1, 2)
# -----------------------------
# Utils: DropPath (stochastic depth)
# -----------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor = random_tensor.div(keep_prob)
        return x * random_tensor

# -----------------------------
# CPE (Conv Positional Encoding) for NCHW
# -----------------------------

class CPE2D(nn.Module):
    """Depthwise 3x3 conv + residual, expect [B, C, H, W]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x) + x

# -----------------------------
# IRB-style FFN in NCHW
# -----------------------------

class IRB2D(nn.Module):
    """ IRB-style MLP on [B, C, H, W]
        1x1 conv -> GELU -> DWConv(3x3) (residual) -> 1x1 conv -> Dropout
    """
    def __init__(self, in_ch: int, hidden_ch: int = None, out_ch: int = None, ksize: int = 3, drop: float = 0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or in_ch
        self.fc1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=ksize, stride=1, padding=ksize // 2, groups=hidden_ch)
        self.fc2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = y + self.dw(y)
        y = self.fc2(y)
        y = self.drop(y)
        return y

# -----------------------------
# Bottleneck Attention in NCHW (spatial attention)
# -----------------------------

class BottleneckAttention2D(nn.Module):
    """
    1x1 conv to reduce channel (C -> C_attn), multi-head attention over spatial tokens (H*W), then 1x1 conv back.
    Input/Output: [B, C, H, W]
    """
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert num_heads > 0, "num_heads must be > 0"
        self.dim = dim
        self.num_heads = num_heads
        self.dim_attn = max(1, int(dim * attn_ratio))
        assert self.dim_attn % num_heads == 0, f"dim_attn ({self.dim_attn}) must be divisible by num_heads ({num_heads})"
        self.head_dim = self.dim_attn // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, self.dim_attn * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.dim_attn, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)  # [B, 3*C_attn, H, W]
        q, k, v = torch.split(qkv, self.dim_attn, dim=1)  # each [B, C_attn, H, W]

        # reshape to heads
        N = H * W
        q = q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # [B, heads, N, head_dim]
        k = k.view(B, self.num_heads, self.head_dim, N)                      # [B, heads, head_dim, N]
        v = v.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # [B, heads, N, head_dim]

        attn = torch.matmul(q * self.scale, k)  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B, heads, N, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.dim_attn, H, W)  # [B, C_attn, H, W]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# -----------------------------
# Modified Spatial Transformer Block (NCHW end-to-end)
# -----------------------------

class Block2D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, attn_ratio: float = 0.5,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,act_layer=nn.GELU,
                 norm_layer: nn.Module = nn.BatchNorm2d):
        super().__init__()
        self.cpe = CPE2D(embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.attn = BottleneckAttention2D(embed_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        hidden_ch = int(embed_dim * mlp_ratio)
        self.ffn = IRB2D(in_ch=embed_dim, hidden_ch=hidden_ch, out_ch=embed_dim, drop=drop)

        # HF refinement: reduce -> DW -> expand (to 3*C)
        self.hf_reduce = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)
        self.hf_dw = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.hf_expand = nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1)
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Positional encoding (CPE in NCHW)
        x = self.cpe(x)

        # Wavelet branch
        LL, HL, LH, HH = dwt2_haar(self.norm1(x))  # norm before splitting

        # Attention on LL (NCHW)
        LL_attn = self.attn(LL)

        # HF refine (concat -> reduce -> DW -> expand -> split)
        HF = torch.cat([HL, LH, HH], dim=1)              # [B, 3C, H/2, W/2]
        HF = self.hf_reduce(HF)
        HF = self.hf_dw(HF)
        HF = self.act(HF)
        HF = self.hf_expand(HF)                          # [B, 3C, H/2, W/2]
        HL_ref, LH_ref, HH_ref = torch.chunk(HF, 3, dim=1)

        # Inverse DWT and residual to x
        x_recon = idwt2_haar(LL_attn, HL_ref, LH_ref, HH_ref)  # [B, C, H, W]
        x = x + self.drop_path(x_recon)

        # FFN on full-res
        y = self.ffn(self.norm2(x))
        x = x + self.drop_path(y)
        return x


class IRB(nn.Module):
    """
    IRB-style MLP:
      1x1 conv -> GELU -> DWConv(3x3) 残差 -> 1x1 conv -> Dropout
    在 [B,C,H,W] 上做，然后再拉回 [B,N,C]
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 ksize=3,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features,
                             kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features,
                                kernel_size=ksize,
                                stride=1,
                                padding=ksize // 2,
                                groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features,
                             kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # x: [B, N, C]
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)   # -> [B,C,H,W]

        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x)                      # DWConv 残差
        x = self.fc2(x)

        x = x.flatten(2).transpose(1, 2)            # -> [B,N,C]
        x = self.drop(x)
        return x

class RetNetRelPos2d(nn.Module):
    """
    2D relative position + exponential decay mask for Retention.

    embed_dim: 总通道 C
    num_heads: 注意力 head 数量

    forward((H, W)) 返回:
      ((sin[H,W,D], cos[H,W,D]), mask[heads, H*W, H*W])
      其中 D = embed_dim / num_heads
    """
    def __init__(self, embed_dim, num_heads, initial_value=1.0, heads_range=3.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        dim_per_head = embed_dim // num_heads
        assert dim_per_head % 2 == 0, "dim_per_head 必须是偶数，用于 RoPE"

        # RoPE 频率: [D/2] -> [D]
        angle = 1.0 / (10000 ** torch.linspace(0, 1, dim_per_head // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()   # [D]

        # 每个 head 一个衰减因子
        h_idx = torch.arange(num_heads, dtype=torch.float32)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * h_idx / num_heads)
        )   # [heads]

        self.register_buffer('angle', angle)   # [D]
        self.register_buffer('decay', decay)   # [heads]

    def generate_2d_decay(self, H, W):
        device = self.decay.device
        index_h = torch.arange(H, device=device)
        index_w = torch.arange(W, device=device)
        # 旧版本 PyTorch 不支持 indexing='ij'，默认就是 ij 顺序
        gh, gw = torch.meshgrid(index_h, index_w)  # [H,W]
        grid = torch.stack([gh, gw], dim=-1).reshape(H * W, 2)  # [HW,2]
        diff = grid[:, None, :] - grid[None, :, :]  # [HW,HW,2]
        dist = diff.abs().sum(dim=-1)  # [HW,HW]
        mask = dist.unsqueeze(0) * self.decay[:, None, None]  # [heads, HW, HW]
        return mask

    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        H, W = slen
        N = H * W
        device = self.decay.device

        index = torch.arange(N, device=device, dtype=self.angle.dtype)
        # [N, D]
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])

        sin = sin.view(H, W, -1)   # [H,W,D]
        cos = cos.view(H, W, -1)   # [H,W,D]
        mask = self.generate_2d_decay(H, W)  # [heads, N, N]

        return (sin, cos), mask


# -----------------------------
# RoPE 辅助函数
# -----------------------------
def rotate_every_two(x):
    """
    x: [B, heads, H, W, D]
    在最后一个维度做 2D 旋转:
      (x1, x2) -> (-x2, x1)
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1)   # [..., D/2, 2]
    return x_rot.flatten(-2)                 # [..., D]

def theta_shift(x, sin, cos):
    """
    x      : [B, heads, H, W, D]
    sin,cos: [H, W, D]（会自动 broadcast 到 B,heads 上）
    """
    while sin.dim() < x.dim():
        sin = sin.unsqueeze(0)   # -> [1,H,W,D] -> [1,1,H,W,D] ...
        cos = cos.unsqueeze(0)
    return x * cos + rotate_every_two(x) * sin

# -----------------------------
# DWConv on BHWC
# -----------------------------
class DWConv2d(nn.Module):
    """Depthwise conv on BHWC tensor."""
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x):
        # x: [B,H,W,C]
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        return x


# -----------------------------
# VisionRetentionChunk: 2D Retention 注意力
# -----------------------------
class VisionRetentionChunk(nn.Module):
    """
    2D Retention on image tokens.

    输入:
      x      : [B,H,W,C]
      rel_pos: ((sin[H,W,D], cos[H,W,D]), mask[heads, H*W, H*W])

    输出:
      y: [B,H,W,C]
    """
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = value_factor

        assert embed_dim % num_heads == 0
        self.key_dim = embed_dim // num_heads
        assert self.key_dim % 2 == 0, "embed_dim/heads 必须为偶数，用于 RoPE"
        self.value_dim = embed_dim * value_factor // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)

        # Local enhancement (LEPE)
        self.lepe = DWConv2d(embed_dim, kernel_size=5, stride=1, padding=2)

        self.out_proj = nn.Linear(embed_dim * value_factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, rel_pos):
        """
        x: [B,H,W,C]
        rel_pos: ((sin,cos), mask)
          sin, cos: [H,W,D]  (D = key_dim)
          mask    : [heads, H*W, H*W]
        """
        B, H, W, C = x.shape
        (sin, cos), mask = rel_pos
        N = H * W
        heads = self.num_heads

        # 1) 线性投影
        q = self.q_proj(x)  # [B,H,W,C]
        k = self.k_proj(x)
        v = self.v_proj(x)  # [B,H,W,C*factor]

        # 2) 拆 head
        q = q.view(B, H, W, heads, self.key_dim).permute(0, 3, 1, 2, 4)  # [B,heads,H,W,D]
        k = k.view(B, H, W, heads, self.key_dim).permute(0, 3, 1, 2, 4)  # [B,heads,H,W,D]
        v = v.view(B, H * W, heads, self.value_dim).permute(0, 2, 1, 3)  # [B,heads,N,Dv]

        # 3) RoPE
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        # 4) 展平成 token
        q = q.reshape(B, heads, N, self.key_dim)  # [B,heads,N,D]
        k = k.reshape(B, heads, N, self.key_dim)  # [B,heads,N,D]

        # 5) Retention 注意力
        attn = torch.matmul(q * self.scaling, k.transpose(-1, -2))  # [B,heads,N,N]
        # mask: [heads,N,N] -> [1,heads,N,N]
        attn = attn + mask.unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B,heads,N,Dv]
        out = out.permute(0, 2, 1, 3).reshape(B, H, W, heads * self.value_dim)  # [B,H,W,C*factor]

        # 6) Local enhancement
        lepe = self.lepe(x)  # [B,H,W,C]
        if self.factor == 1:
            out = out + lepe
        else:
            out[:, :, :, :C] = out[:, :, :, :C] + lepe

        # 7) 输出线性
        out = self.out_proj(out)  # [B,H,W,C]
        return out



class RetNetAngBlock(nn.Module):
    """
    使用 RetNet 相对位置 + VisionRetentionChunk 的角度域 Block。
    输入/输出: x [B, N, C], 其中 N = H*W（例如 k*k 的角度窗口）
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 initial_value=1.0, heads_range=3.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = norm_layer(dim)
        self.retpos = RetNetRelPos2d(embed_dim=dim,
                                     num_heads=num_heads,
                                     initial_value=initial_value,
                                     heads_range=heads_range)
        self.retention = VisionRetentionChunk(embed_dim=dim,
                                              num_heads=num_heads,
                                              value_factor=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = IRB(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       out_features=dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, f"N = {N} 必须是 H*W 的完全平方数"

        # 1) Retention 注意力
        x_norm = self.norm1(x)               # [B,N,C]
        x_hw = x_norm.view(B, H, W, C)       # [B,H,W,C]
        rel_pos = self.retpos((H, W))        # ((sin,cos), mask)
        attn_out = self.retention(x_hw, rel_pos).view(B, N, C)
        x = x + self.drop_path(attn_out)

        # 2) FFN (IRB)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class spa_TransfomerBlock(nn.Module):

    def __init__(self, embed_dim,  spa_num_heads,  spa_mlp_ratio, spa_trans_num, attn_ratio=1, qkv_bias=True,
                  qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer: nn.Module = nn.BatchNorm2d):
        super().__init__()
        # K cascaded spatial transformers
        self.spa_Transformer_Blocks = nn.ModuleList()
        self.spa_trans_num = spa_trans_num
        for i in range(spa_trans_num):
            self.spa_Transformer_Blocks.append(
                Block2D(embed_dim=embed_dim, num_heads=spa_num_heads, mlp_ratio=spa_mlp_ratio, attn_ratio=attn_ratio, drop=drop,
                      attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))

    def forward(self, feature, angRes, patch_size):
        # K cascaded spatial transformers

        for i in range(self.spa_trans_num):
            feature = self.spa_Transformer_Blocks[i](feature)

        return feature


class ang_TransfomerBlock(nn.Module):

    def __init__(self, embed_dim, ang_num_heads, ang_mlp_ratio,
                 ang_trans_num=1,                 # 可堆叠层数
                 attn_ratio=1,                  # 传给 Block 的 sr_ratio
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.ang_trans_num = ang_trans_num

        # 串行堆叠若干 Block（与你的 spa_TransfomerBlock 相同的 Block 定义）
        self.blocks = nn.ModuleList([
            RetNetAngBlock(
                dim=embed_dim,
                num_heads=ang_num_heads,
                mlp_ratio=ang_mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                act_layer=act_layer,
                norm_layer=norm_layer,  # 一般就是 nn.LayerNorm
                initial_value=1.0,
                heads_range=3.0,
            )
            for i in range(ang_trans_num)
        ])

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, k: int):
        """Reflect-pad H,W 到 k 的倍数；返回 (x_pad, (ph, pw), H_old, W_old)"""
        B, C, H, W = x.shape
        ph = (k - (H % k)) % k
        pw = (k - (W % k)) % k
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x, (ph, pw), H, W

    @staticmethod
    def _crop_from_pad(x: torch.Tensor, pad: tuple, H: int, W: int):
        ph, pw = pad
        if ph or pw:
            return x[:, :, :H, :W]
        return x

    def forward(self, feature, angRes, patch_size):
        """
        feature: [B, C, AH, AW]
        angRes : 宏像素的边长（窗口=步长=angRes）
        patch_size: 保留以兼容接口；此实现不依赖它来回排形状
        return : [B, C, AH, AW]
        """
        B, C, AH, AW = feature.shape
        k = int(angRes)  # 宏像素大小

        # 1) pad 到 k 的整数倍，避免边界不整齐
        x, pad, H_old, W_old = self._pad_to_multiple(feature, k)   # [B,C,H',W']

        # 2) 切成不重叠宏像素块：unfold -> [B, C*k*k, L]
        #    其中 L = (H'/k)*(W'/k)
        windows = F.unfold(x, kernel_size=(k, k), stride=(k, k))   # [B, C*k*k, L]

        # 3) 转成 Block 期望的 token 形状：[(B*L), k*k, C]
        #    注意：Block 内部用 sqrt(N)=k 做 sr_ratio>1 的2D重排，这里满足 N=k*k。
        tokens = rearrange(windows, 'b (c p) l -> (b l) p c', c=C, p=k*k)

        # 4) 逐层自注意力
        for blk in self.blocks:
            tokens = blk(tokens)  # [(B*L), k*k, C]

        # 5) 回到卷积形状并 fold 回去
        windows_out = rearrange(tokens, '(b l) p c -> b (c p) l', b=B, c=C, p=k*k)
        y_pad = F.fold(windows_out, output_size=(x.shape[-2], x.shape[-1]),
                       kernel_size=(k, k), stride=(k, k))         # [B,C,H',W']

        # 6) 裁回原尺寸
        y = self._crop_from_pad(y_pad, pad, H_old, W_old)          # [B,C,AH,AW]
        return y




class SConv(nn.Module):
    def __init__(self, in_channels, out_channels, angRes):
        super(SConv, self).__init__()
        self.angRes = angRes
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes,
                               bias=False)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes,
                               bias=False)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class AConv(nn.Module):
    def __init__(self, in_channels, out_channels, angRes):
        super(AConv, self).__init__()
        self.angRes = angRes
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=angRes, stride=angRes, padding=0, bias=False)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

def tile_ang_to_sai(ang_feat, B, angRes):
    # ang_feat: (B, C, H, W) -> (B*a*a, C, H, W) 不拷贝数据，用 expand
    return (ang_feat.unsqueeze(1)  # (B,1,C,H,W)
                   .expand(B, angRes*angRes, ang_feat.size(1), ang_feat.size(2), ang_feat.size(3))
                   .reshape(B * (angRes**2), ang_feat.size(1), ang_feat.size(2), ang_feat.size(3)))


def make_dpr_taker(dpr_list):
    idx = 0

    def take(n):
        nonlocal idx
        out = dpr_list[idx: idx + n]
        assert len(out) == n, f"drop_path slice len {len(out)} != expected {n} at idx={idx}"
        idx += n
        return out

    return take

class EnhancedFusionBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        """
        channels: 输入单个分支的通道数，两个分支拼接后为 channels*2
        reduction: SE注意力的缩放比例
        """
        super().__init__()
        # 1. 初始卷积 + 非线性
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        # 2. 第二卷积 + 残差
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        # 3. SE 注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 4. 输出卷积
        self.conv3 = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # x: [B, 2*C, H, W]
        out1 = self.act1(self.conv1(x))
        out2 = self.act2(self.conv2(out1))
        out2 = out2 + out1  # 残差连接
        out2 = out2 * self.se(out2)  # SE 注意力
        out = self.conv3(out2)
        return out

class LF_WXformerCHUAN(nn.Module):
    """
    串联级联消融版：
      SAI -> 空间 Transformer -> MPI -> 角度 Transformer -> SAI
      如此交替级联 5 次，不再使用 BCU 中的多层双向残差互馈 (SpaConv/AngConv/alpha/beta)。
    用于与原始并行 + 双向交互结构做对比。
    """
    def __init__(self, args, cascade_depth: int = 5):
        super(LF_WXformerCHUAN, self).__init__()
        self.channels = args.channels
        self.angRes = args.angRes

        self.patch_size = args.patch_size
        self.num_ang = self.angRes * self.angRes
        self.num_spa = self.patch_size * self.patch_size

        self.cascade_depth = cascade_depth  # 级联层数，默认为 5

        # step1: Local Feature Extraction（和原网络一致）
        self.conv_init0 = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=3, padding=1, dilation=1, bias=False)
        )
        self.conv_init_spa = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

        spatial_fixed_depth = 4
        ang_fixed_depth = 4

        def make_linear_dpr(n, rate):
            if n <= 0:
                return []
            return [float(x) for x in torch.linspace(0.0, float(rate), steps=n)]

        dpr_spa = make_linear_dpr(spatial_fixed_depth, args.drop_path_rate)
        dpr_ang = make_linear_dpr(ang_fixed_depth, args.drop_path_rate)

        # ---- 空间分支：与原始网络相同 ----
        self.spa = spa_TransfomerBlock(
            self.channels,
            args.spa_num_heads,
            args.spa_mlp_ratio,
            spa_trans_num=spatial_fixed_depth,
            attn_ratio=args.attn_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop=args.drop_rate,
            attn_drop=args.attn_drop_rate,
            drop_path=dpr_spa,
            norm_layer=LayerNorm2d
        )

        # ---- 角度分支：与原始网络相同 ----
        self.ang = ang_TransfomerBlock(
            self.channels,
            args.ang_num_heads,
            args.ang_mlp_ratio,
            ang_trans_num=ang_fixed_depth,
            qkv_bias=True,
            qk_scale=None,
            drop=args.drop_rate,
            attn_drop=args.attn_drop_rate,
            drop_path=dpr_ang,
            norm_layer=nn.LayerNorm
        )

        # 最终融合与重建（保持不变）
        self.output = EnhancedFusionBlock(self.channels)

    def forward(self, inp_img, data_info=None):
        """
        串联流程 (交替 5 层)：
          1) SAI -> Local feature
          2) for i in 1..cascade_depth:
               SAI -> spa -> MPI -> ang -> SAI
          3) 最后一次的空间特征 & 角度特征拼接, 通过 EnhancedFusionBlock 输出，再加残差
        """
        b, u, v, c, h, w = inp_img.size()

        # ---------- reshape 到 SAI 并做局部特征提取 ----------
        # inp_img: [B, U, V, C, H, W]
        inp = inp_img.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, u * h, v * w)
        inp = rearrange(inp, 'b c (u h) (v w) -> b c (u v) h w', u=self.angRes, v=self.angRes)
        inp = rearrange(inp, 'b c a h w -> (b a) c h w')   # [B*A^2, C, Hs, Ws]

        # Local feature
        init_feature = self.conv_init0(inp)
        init_feature = self.conv_init_spa(init_feature) + init_feature   # [B*A^2, C, Hs, Ws]

        # 接下来 Hs, Ws 固定为 patch_size
        tokens = rearrange(init_feature.flatten(2), 'b c hw -> b hw c')
        feature = rearrange(tokens, 'b (h w) c -> b c h w',
                            h=self.patch_size, w=self.patch_size)      # 作为 SAI 特征

        # ---------- 串联级联 5 层：SAI -> spa -> MPI -> ang -> SAI ----------
        spa_feat_last = None
        ang_mpi_last = None

        for _ in range(self.cascade_depth):
            # 1) SAI 上做空间 Transformer
            spa_feat = self.spa(feature, self.angRes, self.patch_size)      # [B*A^2, C, Hs, Ws]

            # 2) SAI -> MPI，进入角度分支
            mpi_feat = rearrange(
                spa_feat, '(b u v) c h w -> b c (h u) (w v)',
                u=self.angRes, v=self.angRes
            )  # [B, C, Hs*u, Ws*v]

            # 3) MPI 上做角度 Transformer
            ang_mpi = self.ang(mpi_feat, self.angRes, self.patch_size)      # [B, C, Hs*u, Ws*v]

            # 4) MPI -> SAI，作为下一层级联的输入
            feature = rearrange(
                ang_mpi, 'b c (h u) (w v) -> (b u v) c h w',
                u=self.angRes, v=self.angRes
            )  # [B*A^2, C, Hs, Ws]

            spa_feat_last = spa_feat
            ang_mpi_last = ang_mpi

        # ---------- 最终融合与重建 ----------
        # 角度分支输出：从最后一次的 MPI 还原到每个视图
        out_dec_level1 = rearrange(
            ang_mpi_last, 'b c (h u) (w v) -> (b u v) c h w',
            u=self.angRes, v=self.angRes
        )  # [B*A^2, C, Hs, Ws]

        # 空间分支输出：最后一次 spa_feat
        out_spa = spa_feat_last                                       # [B*A^2, C, Hs, Ws]

        x = torch.cat([out_dec_level1, out_spa], dim=1)               # [B*A^2, 2C, Hs, Ws]

        buffer = self.output(x)                                       # [B*A^2, 3, Hs, Ws]

        buffer = rearrange(buffer, '(b u v) c h w -> b (u v) c h w',
                           u=self.angRes, v=self.angRes)
        final_buffer = rearrange(
            buffer, 'b (u v) c h w -> b u v c h w',
            u=self.angRes, v=self.angRes
        )

        out = final_buffer + inp_img
        return out


def interpolate(x, angRes, scale_factor, mode):
    [B, _, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, 1, angRes, h, angRes, w)
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor, W * scale_factor)

    return x_upscale


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


def weights_init(m):
    pass


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import args
    from fvcore.nn import FlopCountAnalysis

    net = LF_WXformerCHUAN(args).cuda()
    input = torch.randn(1,5,5,3,32,32).cuda()
    flops = FlopCountAnalysis(net, input)
    print("Flops: ", flops.total())
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.4fM" % (total / 1e6))