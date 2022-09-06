from typing import Union
import argparse
import os
import os.path as osp
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from data import get_jnp_dataset
import ml_collections
from config import get_config
from models import (
    create_PVT_V2,
    PVT_V2_B0,
    PVT_V2_B1,
    PVT_V2_B2,
    PVT_V2_B3,
    PVT_V2_B4,
    PVT_V2_B5,
)
from utils import save_checkpoint
from termcolor import colored

model_dict = {
    "PVT_V2_B0": PVT_V2_B0,
    "PVT_V2_B1": PVT_V2_B1,
    "PVT_V2_B2": PVT_V2_B2,
    "PVT_V2_B3": PVT_V2_B3,
    "PVT_V2_B4": PVT_V2_B4,
    "PVT_V2_B5": PVT_V2_B5,
}


def apply_model_non_jit(state, images, labels, trainable, rng):
    """
    Computes gradients, loss and accuracy for a single batch.
    """
    
    def loss_fn(params, trainable, rng):
        logits = state.apply_fn(
            {"params": params}, images, trainable=trainable, rngs={"dropout": rng}
        )
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, trainable, rng)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    return grads, loss, accuracy

apply_model = jax.jit(apply_model_non_jit, static_argnums=(3,))

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """
    Train for a single epoch.
    """
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model(
            state, batch_images, batch_labels, True, rng
        )
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)

    return state, train_loss, train_accuracy


def create_train_state(
    model_name: str, num_classes: int, cfg: ml_collections.ConfigDict
):
    """
    Creates initial `TrainState`. For more information
        refer to the official Flax documentation.
    https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state
    """
    model, params = create_PVT_V2(model_dict[model_name], num_classes=num_classes)
    tx = optax.adamw(learning_rate=cfg.learning_rate)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_and_evaluate(
    state: train_state.TrainState,
    cfg: ml_collections.ConfigDict,
    dataset_name: str,
    work_dir: Union[os.PathLike, str],
    train_ds,
    test_ds,
) -> train_state.TrainState:
    """
    Execute model training and evaluation loop

    Returns:
      The train state (which includes the updated
        parameters that can be accessed with 'state.params`).
    """
    os.makedirs(osp.join(work_dir, "logs"), exist_ok=True)
    summary_writer = tensorboard.SummaryWriter(osp.join(work_dir, "logs"))
    summary_writer.hparams(dict(cfg))
    rng = random.PRNGKey(0)

    for epoch in range(0, cfg.num_epochs):
        rng, input_rng = random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, cfg.batch_size, input_rng
        )
        save_checkpoint(state=state, epoch=epoch, output_dir=work_dir)
        _, test_loss, test_accuracy = apply_model(
            state, test_ds["image"], test_ds["label"], False, rng
        )

        print(colored(f"Epoch: {epoch}", "cyan"))
        print(
            "train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )

        summary_writer.scalar("Train_Loss", train_loss, epoch)
        summary_writer.scalar("Train_Accuracy", train_accuracy, epoch)
        summary_writer.scalar("Test_Loss", test_loss, epoch)
        summary_writer.scalar("Test_Accuracy", test_accuracy, epoch)

    summary_writer.flush()
    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Evaluate your Flax model.")
    parser.add_argument(
        "--model-name",
        help="Name of the model variant.",
        required=True,
    )
    parser.add_argument(
        "--work-dir",
        help="Path where TensorBoard events and checkpoints save.",
        default="output/",
        required=False,
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of the dataset. Must be implemented in TFDS.",
        required=True,
    )
    parser.add_argument(
        "--num-classes",
        help="Number of classes in the dataset.",
        required=True,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    assert (
        args.model_name in model_dict
    ), f"Method {args.model_name} not yet implemented."

    cfg = get_config()
    train_ds, test_ds = get_jnp_dataset(name=args.dataset_name)
    state = create_train_state(args.model_name, num_classes=int(args.num_classes), cfg=cfg)

    train_and_evaluate(state, cfg, args.dataset_name, args.work_dir, train_ds, test_ds)
