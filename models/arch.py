from functools import partial
import jax.numpy as jnp
import numpy as np
from jax import random
import flax.linen as nn
from typing import Union, Iterable, Callable, List, Optional
from layers import (
    DropPath,
    to_ntuple,
    Attention,
    OverlapPatchEmbed,
    MLP_with_DepthWiseConv,
)
from utils import restore_checkpoint
from flax.core import freeze, unfreeze


class Block(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    sr_ratio: int = 1
    linear: bool = False
    qkv_bias: bool = False
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    act: Callable = nn.gelu
    norm_layer: Callable = nn.LayerNorm
    trainable: Optional[bool] = None

    @nn.compact
    def __call__(self, x, feat_size=List[int], trainable=None):
        trainable = nn.merge_param("trainable", self.trainable, trainable)
        norm = self.norm_layer()
        attn = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            sr_ratio=self.sr_ratio,
            linear=self.linear,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            trainable=trainable,
        )

        mlp = MLP_with_DepthWiseConv(
            hidden_features=int(self.dim * self.mlp_ratio),
            act=self.act,
            drop=self.drop,
            extra_relu=self.linear,
            trainable=trainable,
        )

        if self.drop_path > 0.0:
            drop_path = DropPath(self.drop_path, trainable=not trainable)
            x = x + drop_path(attn(norm(x), feat_size))
            x = x + drop_path(mlp(norm(x), feat_size))

        else:
            x = x + attn(norm(x), feat_size)
            x = x + mlp(norm(x), feat_size)

        return x


class PyramidVisionTransformerStage(nn.Module):
    dim_out: int
    depth: int
    downsample: bool = True
    num_heads: int = 8
    mlp_ratio: float = 4.0
    sr_ratio: int = 1
    linear: bool = True
    qkv_bias: bool = True
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: Union[List[float], float] = 0.0
    norm_layer: Callable = nn.LayerNorm
    trainable: Optional[bool] = None

    @nn.compact
    def __call__(self, x, feat_size: List[int], trainable=None):
        trainable = nn.merge_param("trainable", self.trainable, trainable)
        norm = self.norm_layer()
        dim = x.shape[-1]

        if self.downsample:
            downsample = OverlapPatchEmbed(
                patch_size=3, strides=2, embed_dim=self.dim_out
            )
            x, feat_size = downsample(x)

        else:
            assert (
                dim == self.dim_out
            ), "With downsample=False, input dims should be equal to dim_out."

        for i in range(self.depth):
            x = Block(
                dim=self.dim_out,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                sr_ratio=self.sr_ratio,
                linear=self.linear,
                qkv_bias=self.qkv_bias,
                drop=self.drop,
                attn_drop=self.attn_drop,
                drop_path=self.drop_path[i]
                if isinstance(self.drop_path, list)
                else self.drop_path,
                norm_layer=self.norm_layer,
                trainable=trainable,
            )(x, feat_size)

        x = norm(x)
        x = x.reshape(x.shape[0], feat_size[0], feat_size[1], -1)

        return x, feat_size


class PyramidVisionTransformerV2(nn.Module):
    img_size: Union[int, Iterable[int]] = None
    depths: Iterable[int] = (3, 4, 6, 3)
    embed_dims: Iterable[int] = (64, 128, 256, 512)
    num_heads: Iterable[int] = (1, 2, 4, 8)
    sr_ratios: Iterable[int] = (8, 4, 2, 1)
    mlp_ratios: Iterable[int] = (8.0, 8.0, 4.0, 4.0)
    qkv_bias: bool = True
    linear: bool = False
    drop_rate: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    norm_layer: Callable = nn.LayerNorm
    attach_head: bool = True
    num_classes: int = 1000
    trainable: Optional[bool] = None

    @nn.nowrap
    def classify(self, x, head: Callable, pre_logits: bool = False):
        x = jnp.mean(x, axis=(1, 2))
        return x if pre_logits else head(x)

    @nn.compact
    def __call__(self, x, trainable=None):
        trainable = nn.merge_param("trainable", self.trainable, trainable)
        assert not (
            self.attach_head and self.num_classes == 0
        ), f"If attach_head=True, num_classes should be greater than zero."
        num_stages = len(self.depths)
        mlp_ratios = to_ntuple(num_stages)(self.mlp_ratios)
        num_heads = to_ntuple(num_stages)(self.num_heads)
        sr_ratios = to_ntuple(num_stages)(self.sr_ratios)
        assert (len(self.embed_dims)) == num_stages

        # assert (
        #     x.shape[-1] == 3
        # ), f"Expected an RGB image with 3 channels but got {x.shape[-1]} channels instead."
        patch_embed = OverlapPatchEmbed(
            patch_size=7, strides=4, embed_dim=self.embed_dims[0]
        )

        dpr = [x.item() for x in np.linspace(0, 0.0, sum(self.depths))]
        cur = 0

        x, feat_size = patch_embed(x)

        out_list = []
        for i in range(num_stages):
            x, feat_size = PyramidVisionTransformerStage(
                dim_out=self.embed_dims[i],
                depth=self.depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                sr_ratio=sr_ratios[i],
                mlp_ratio=mlp_ratios[i],
                linear=self.linear,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop,
                drop_path=dpr[i],
                norm_layer=self.norm_layer,
                trainable=trainable,
            )(x, feat_size)
            out_list.append(x)
            cur += self.depths[i]

        if not self.attach_head:
            return out_list

        head = nn.Dense(self.num_classes)
        x = self.classify(x, head)

        return x


def create_PVT_V2(
    model,
    rng,
    attach_head=True,
    num_classes=1000,
    drop_rate=0.0,
    checkpoint=None,
    in_shape=(1, 32, 32, 3),
):
    key, drop = random.split(rng)
    model = model(attach_head=attach_head, num_classes=num_classes, drop_rate=drop_rate)

    if checkpoint:
        pretrained_weights = restore_checkpoint(checkpoint_dir=checkpoint)
        params = model.init(
            {"params": key, "dropout": drop}, jnp.ones(in_shape), trainable=True
        )
        params = unfreeze(params)
        params["params"].update(pretrained_weights)
        params = freeze(params)
    else:
        params = model.init(
            {"params": key, "dropout": drop}, jnp.ones(in_shape), trainable=True
        )

    return model, params["params"]


PVT_V2_B0 = partial(
    PyramidVisionTransformerV2,
    depths=(2, 2, 2, 2),
    embed_dims=(64, 128, 320, 512),
    num_heads=(1, 2, 5, 8),
)

PVT_V2_B1 = partial(
    PyramidVisionTransformerV2,
    depths=(2, 2, 2, 2),
    embed_dims=(64, 128, 320, 512),
    num_heads=(1, 2, 5, 8),
)

PVT_V2_B2 = partial(
    PyramidVisionTransformerV2,
    depths=(3, 4, 6, 3),
    embed_dims=(64, 128, 320, 512),
    num_heads=(1, 2, 5, 8),
)

PVT_V2_B3 = partial(
    PyramidVisionTransformerV2,
    depths=(3, 4, 18, 3),
    embed_dims=(64, 128, 320, 512),
    num_heads=(1, 2, 5, 8),
)

PVT_V2_B4 = partial(
    PyramidVisionTransformerV2,
    depths=(3, 8, 27, 3),
    embed_dims=(64, 128, 320, 512),
    num_heads=(1, 2, 5, 8),
)

PVT_V2_B5 = partial(
    PyramidVisionTransformerV2,
    depths=(3, 6, 40, 3),
    embed_dims=(64, 128, 320, 512),
    num_heads=(1, 2, 5, 8),
)
