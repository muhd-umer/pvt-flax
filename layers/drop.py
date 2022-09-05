import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from typing import Optional


class DropPath(nn.Module):
    dropout_prob: float = 0.1

    @nn.compact
    def __call__(self, x, trainable=False):
        if trainable:
            return x
        keep_prob = 1 - self.dropout_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.make_rng("drop_path")
        random_tensor = keep_prob + random.uniform(rng, shape)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(x, keep_prob) * random_tensor
