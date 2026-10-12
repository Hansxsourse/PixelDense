import torch
import torch.nn as nn
import numpy as np

from functools import lru_cache
from src.models.layers.attention_op import attention
from src.models.layers.rope import apply_rotary_emb, precompute_freqs_cis_ex2d as precompute_freqs_cis_2d
from src.models.layers.time_embed import TimestepEmbedder as TimestepEmbedder
from src.models.layers.patch_embed import Embed as Embed
from src.models.layers.swiglu import SwiGLU as FeedForward
from src.models.layers.rmsnorm import RMSNorm as Norm

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generate 2D sin-cos positional embedding.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_x = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.kv_y = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y, pos) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q.contiguous())
        kx = self.k_norm(kx.contiguous())
        q, kx = apply_rotary_emb(q, kx, freqs_cis=pos)
        kv_y = self.kv_y(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ky, vy = kv_y[0], kv_y[1]
        ky = self.k_norm(ky.contiguous())

        k = torch.cat([kx, ky], dim=2)
        v = torch.cat([vx, vy], dim=2)

        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()

        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4, ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, y, c, pos):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), y, pos)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class TextRefineAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()
        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4, ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = TextRefineAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class BottleneckEmbed(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            bottleneck_dim: int = 256,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj1 = nn.Linear(in_chans, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, embed_dim, bias=bias)

    def forward(self, x):
        x = self.proj2(self.proj1(x))
        return x

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

class JiT_T2I(nn.Module):
    def __init__(
            self,
            input_size=256,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            num_blocks=18,
            num_text_blocks=4,
            patch_size=2,
            txt_embed_dim=1024,
            txt_max_length=100,
            bottleneck_dim=256,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.num_text_blocks = num_text_blocks
        self.patch_size = patch_size
        self.txt_embed_dim = txt_embed_dim
        self.txt_max_length = txt_max_length
        self.x_embedder = BottleneckEmbed(in_channels*patch_size**2, hidden_size, bottleneck_dim, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = Embed(txt_embed_dim, hidden_size, bias=True, norm_layer=Norm)
        
        # Image Positional Embedding (Fixed Sin-Cos)
        self.num_patches = (input_size//patch_size)**2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        
        self.y_pos_embedding = torch.nn.Parameter(
            torch.randn(1, txt_max_length, hidden_size)* 0.02,
            requires_grad=True
        )
        self.visual_type_embedding = nn.Parameter(
            torch.zeros(1, hidden_size),
            requires_grad=True
        )

        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_blocks)
        ])
        
        self.text_refine_blocks = nn.ModuleList([
            TextRefineBlock(self.hidden_size, self.num_groups) for _ in range(self.num_text_blocks)
        ])
        self.final_layer = FinalLayer(self.hidden_size, in_channels*patch_size**2)

        self.initialize_weights()
        self.precompute_pos = dict()
        self.weight_path = weight_path
        self.load_ema = load_ema

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize fixed sin-cos pos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))    
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
    

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.visual_type_embedding, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for block in self.text_refine_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, H, W):
        """
        x: (N, T, patch_size**2 * C)
        H, W: original height and width
        """
        x = torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(), 
            output_size=(H, W), 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        return x
    
    def forward(self, x, t, y, return_layer=None):
        B, _, H, W = x.shape
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        xpos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        ypos = self.y_pos_embedding
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(B, -1, self.hidden_size) + ypos.to(y.dtype)
        condition = nn.functional.silu(t)
        for i, block in enumerate(self.text_refine_blocks):
            y = block(y, condition)

        x = self.x_embedder(x)
        x = x + self.pos_embed
        x = x + self.visual_type_embedding
        for i, block in enumerate(self.blocks):
            if return_layer is not None and i==return_layer:
                feat = x
            x = block(x, y, condition, xpos)
            
        x = self.final_layer(x)
        output = self.unpatchify(x, H, W)
        if return_layer is not None:
            return output, feat
        else:
            return output