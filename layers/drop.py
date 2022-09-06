import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from typing import Optional


class DropPath(nn.Module):
    dropout_prob: float = 0.1
    trainable: Optional[bool] = None

    @nn.compact
    def __call__(self, x, trainable=None):
        trainable = nn.merge_param("trainable", self.trainable, trainable)
        if trainable:
            return x
        keep_prob = 1 - self.dropout_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + random.uniform(random.PRNGKey(0), shape)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(x, keep_prob) * random_tensor
