import jax.numpy as jnp
import flax.linen as nn
from typing import List, Optional
from termcolor import colored
from .pool import AdaptiveAveragePool2D


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    sr_ratio: int = 1
    linear: bool = False
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    trainable: Optional[bool] = None

    @nn.compact
    def __call__(self, x, feat_size=List[int], trainable=None):
        trainable = nn.merge_param("trainable", self.trainable, trainable)
        assert self.dim % self.num_heads == 0, colored(
            f"Input dim {self.dim} should be divisible by num_heads {self.num_heads}.",
            "red",
        )
        head_dim = self.dim // self.num_heads
        scale = head_dim**-0.5

        attn_drop = nn.Dropout(self.attn_drop, deterministic=not trainable)
        proj = nn.Dense(self.dim)
        proj_drop = nn.Dropout(self.proj_drop, deterministic=not trainable)

        B, N, C = x.shape
        H, W = feat_size

        q = nn.Dense(self.dim, use_bias=self.qkv_bias)(x)
        q = jnp.transpose(q.reshape(B, N, self.num_heads, -1), (0, 2, 1, 3))

        if not self.linear:
            pool = None
            if self.sr_ratio > 1:
                sr = nn.Conv(
                    self.dim,
                    kernel_size=(self.sr_ratio, self.sr_ratio),
                    strides=self.sr_ratio,
                )
                norm = nn.LayerNorm()
            else:
                sr = None
                norm = None
            act = None
        else:
            pool = AdaptiveAveragePool2D(7)
            sr = nn.Conv(self.dim, kernel_size=(1, 1), strides=1)
            norm = nn.LayerNorm()
            act = nn.gelu

        if pool is not None:
            x_ = x.reshape(B, H, W, C)
            x_ = sr((pool(x_))).reshape(B, -1, C)
            x_ = norm(x_)
            x_ = act(x_)
            kv = nn.Dense(self.dim * 2, use_bias=self.qkv_bias)(x_)
            kv = jnp.transpose(
                kv.reshape(B, -1, 2, self.num_heads, head_dim), (2, 0, 3, 1, 4)
            )
        else:
            if sr is not None:
                x_ = x.reshape(B, H, W, C)
                x_ = sr(x_).reshape(B, -1, C)
                x_ = norm(x_)
                kv = nn.Dense(self.dim * 2, use_bias=self.qkv_bias)(x_)
                kv = jnp.transpose(
                    kv.reshape(B, -1, 2, self.num_heads, head_dim), (2, 0, 3, 1, 4)
                )
            else:
                kv = nn.Dense(self.dim * 2, use_bias=self.qkv_bias)(x)
                kv = jnp.transpose(
                    kv.reshape(B, -1, 2, self.num_heads, head_dim), (2, 0, 3, 1, 4)
                )

        k, v = kv[0], kv[1]

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = nn.softmax(attn, axis=-1)
        attn = attn_drop(attn)

        x = jnp.swapaxes(attn @ v, 1, 2).reshape(B, N, C)
        x = proj(x)
        x = proj_drop(x)

        return x
