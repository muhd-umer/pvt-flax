import os
from typing import Iterable, Union
import numpy as np
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
from termcolor import colored
import numpy as np
from termcolor import colored
from clu import parameter_overview


class FlattenAndCast(object):
    """
    Returns contigious flattened array to make Numpy arrays.
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """

    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


class ReshapeAndCast(object):
    """
    Returns contigious flattened array to make Numpy arrays.
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """

    def __call__(self, pic):
        return np.reshape(np.array(pic, dtype=jnp.float32), -1)


def numpy_collate(batch):
    """
    Collate function to use PyTorch datalaoders
        with JAX/Flax.
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def save_checkpoint(
    target: train_state.TrainState, epoch: int, output_dir: Union[os.PathLike, str]
):
    """
    Args:
        target: TrainState to save as a checkpoint
        epoch: Training step number
        output_dir: Directory to save checkpoints to.
    Returns:
        None
    """
    save_dir = checkpoints.save_checkpoint(
        ckpt_dir=str(output_dir), target=target, step=epoch, overwrite=True
    )
    print(colored(f"{' '*10} Saving checkpoint at {save_dir}", "magenta"))


def restore_checkpoint(checkpoint_dir: Union[os.PathLike, str]):
    """
    Args:
        checkpoint_dir: Directory to load checkpoint from.
    Returns:
        None
    """
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir, target=train_state.TrainState
    )
    print(colored(f"Restoring state from {checkpoint_dir}", "magenta"))
    return restored_state.params


def tabulate(model: nn.Module, input_shape: Iterable[int] = (1, 32, 32, 3)):
    key = random.PRNGKey(0)
    variables = model.init(key, jnp.ones(input_shape))

    print(colored(parameter_overview.get_parameter_overview(variables), "cyan"))
