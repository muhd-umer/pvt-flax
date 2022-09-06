import flax.linen as nn
from typing import Callable, List, Optional


class MLP_with_DepthWiseConv(nn.Module):
    hidden_features: int = None
    out_features: int = None
    act: Callable = nn.gelu
    drop: float = 0.0
    extra_relu: bool = False
    trainable: Optional[bool] = None

    @nn.compact
    def __call__(self, x, feat_size=List[int], trainable=None):
        trainable = nn.merge_param("trainable", self.trainable, trainable)
        in_features = x.shape[-1]
        out_features = self.out_features or in_features
        hidden_features = self.hidden_features or in_features
        drop = nn.Dropout(rate=self.drop, deterministic=not trainable)

        x = nn.Dense(hidden_features)(x)
        B, N, C = x.shape
        H, W = feat_size
        x = x.reshape(B, H, W, C)
        x = nn.relu(x)
        x = nn.Conv(
            hidden_features,
            kernel_size=(3, 3),
            use_bias=True,
            feature_group_count=hidden_features,
        )(x)
        x = x.reshape(B, -1, x.shape[3])
        x = self.act(x)
        x = drop(x)
        x = nn.Dense(out_features)(x)
        x = drop(x)

        return x
