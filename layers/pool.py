import jax.numpy as jnp
import flax.linen as nn
from typing import Union, Iterable
from .helpers import to_2tuple


class AdaptiveAveragePool2D(nn.Module):
    output_size: Union[int, Iterable[int]]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert (
            x.ndim > 2 and x.ndim < 5
        ), f"Expected input to have 3 (or 4 with batch) but recieved {x.ndim} dims."
        if x.ndim == 4:
            input_size = x.shape[1], x.shape[2]
        else:
            input_size = x.shape[0], x.shape[1]
        if not isinstance(self.output_size, Iterable):
            output_size = to_2tuple(self.output_size)
        else:
            output_size = self.output_size

        strides = tuple(e1 // e2 for e1, e2 in zip(input_size, output_size))
        factor = tuple((e1 - 1) * e2 for e1, e2 in zip(output_size, strides))
        window_shape = tuple(e1 - e2 for e1, e2 in zip(input_size, factor))
        x = nn.avg_pool(x, window_shape=window_shape, strides=strides, padding="VALID")

        return x
