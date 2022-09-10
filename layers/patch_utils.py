import flax.linen as nn
from termcolor import colored
from .helpers import to_2tuple


class OverlapPatchEmbed(nn.Module):
    patch_size: int = 7
    strides: int = 4
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x):
        patch_size = to_2tuple(self.patch_size)
        assert max(patch_size) > self.strides, colored(
            "Patch size should be larger than stride.", "red"
        )
        norm = nn.LayerNorm()

        x = nn.Conv(
            self.embed_dim,
            kernel_size=patch_size,
            strides=self.strides,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )(x)
        feat_size = x.shape[1:3]
        x = x.reshape(x.shape[0], -1, x.shape[3])
        x = norm(x)

        return x, feat_size
