from typing import Union, Iterable
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import os.path as osp
import numpy as np
from utils import save_checkpoint, restore_checkpoint
from termcolor import colored
import time

import jax
from jax import random
import jax.numpy as jnp
import optax
from flax.metrics import tensorboard
from flax.training import train_state
from flax.core import freeze, unfreeze
import tensorflow as tf
from clu import platform

from data import get_jnp_dataset
from tqdm import tqdm
from config import get_config
from models import (
    PVT_V2_B0,
    PVT_V2_B1,
    PVT_V2_B2,
    PVT_V2_B3,
    PVT_V2_B4,
    PVT_V2_B5,
)

model_dict = {
    "PVT_V2_B0": PVT_V2_B0,
    "PVT_V2_B1": PVT_V2_B1,
    "PVT_V2_B2": PVT_V2_B2,
    "PVT_V2_B3": PVT_V2_B3,
    "PVT_V2_B4": PVT_V2_B4,
    "PVT_V2_B5": PVT_V2_B5,
}


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def apply_model(state, inputs, labels, num_classes, dropout_rng=None, trainable=False):
    """
    Defines a single apply on model.
    Returns:
        grads: To apply to the state
        loss: Mean loss for the current batch
        accuracy: Mean accuracy for the current batch
    """
    if dropout_rng:
        dropout_rng = random.fold_in(dropout_rng[0], state.step)

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                inputs,
                trainable=trainable,
                rngs={"dropout": dropout_rng},
            )
            one_hot = jax.nn.one_hot(labels, num_classes)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss, logits

    else:
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                inputs,
                trainable=trainable,
            )
            one_hot = jax.nn.one_hot(labels, num_classes)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    return grads, loss, accuracy


apply_model = jax.jit(apply_model, static_argnums=(3, 5))


def train_epoch(state, train_ds, num_classes, total, dropout_rng):
    """
    Defines a single training epoch (forward passes).
    Returns:
        state: Updated state of the model
        loss: Mean loss for the current batch
        accuracy: Mean accuracy for the current batch
    """
    train_loss, train_accuracy = list(), list()

    for batch in tqdm(
        train_ds,
        total=total,
        desc=colored(f"{' '*10} Training", "magenta"),
        colour="cyan",
    ):
        inputs, labels = batch["image"], batch["label"]
        inputs = jnp.float32(inputs) / 255.0
        labels = jnp.float32(labels)

        grads, loss, accuracy = apply_model(
            state, inputs, labels, num_classes, dropout_rng, True
        )
        state = update_model(state, grads)
        train_loss.append(loss)
        train_accuracy.append(accuracy)

    epoch_loss = np.mean(train_loss)
    epoch_accuracy = np.mean(train_accuracy)

    return state, epoch_loss, epoch_accuracy


def test_epoch(state, test_ds, num_classes, total):
    """
    Defines a single test epoch (validation).
    Returns:
        loss: Mean loss for the current batch
        accuracy: Mean accuracy for the current batch
    """
    test_loss, test_accuracy = list(), list()

    for batch in tqdm(
        test_ds,
        total=total,
        desc=colored(f"{' '*10} Validating", "magenta"),
        colour="cyan",
    ):
        inputs, labels = batch["image"], batch["label"]
        inputs = jnp.float32(inputs) / 255.0
        labels = jnp.float32(labels)

        _, loss, accuracy = apply_model(state, inputs, labels, num_classes)
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    epoch_loss = np.mean(test_loss)
    epoch_accuracy = np.mean(test_accuracy)

    return epoch_loss, epoch_accuracy


def create_train_state(
    model,
    rng,
    learning_rate,
    num_classes: int,
    image_shape: Iterable[int],
):
    """
    Creates initial `TrainState`. For more information
        refer to the official Flax documentation.
    https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state
    """

    model = model(num_classes=num_classes)
    params = model.init(rng, jnp.ones(image_shape), False)["params"]

    tx = optax.adamw(learning_rate=learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    return state


create_train_state = jax.jit(create_train_state, static_argnums=(0, 2, 3, 4))


def train_and_evaluate(
    state: train_state.TrainState,
    epochs: int,
    work_dir: Union[os.PathLike, str],
    train_ds,
    total_train,
    test_ds,
    total_test,
    num_classes,
    rng,
) -> train_state.TrainState:
    """
    Execute model training and evaluation loop

    Returns:
      The train state (which includes the updated
        parameters that can be accessed with 'state.params`).
    """
    os.makedirs(osp.join(work_dir, "logs"), exist_ok=True)
    summary_writer = tensorboard.SummaryWriter(osp.join(work_dir, "logs"))
    dropout_rngs = random.split(rng, jax.local_device_count())

    for epoch in range(1, epochs + 1):

        named_tuple = time.localtime()
        time_string = time.strftime("%H:%M:%S", named_tuple)

        print(colored(f"[{time_string}] Epoch: {epoch}", "cyan"))

        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, num_classes, total_train, dropout_rngs
        )

        test_loss, test_accuracy = test_epoch(state, test_ds, num_classes, total_test)

        print(
            f"{' '*10} train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )
        save_checkpoint(target=state, epoch=epoch, output_dir=work_dir)

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
        default="PVT_V2_B0",
        help="Name of the model variant. Currently supports PVT_V2_B[Index] where Index ranges from 0 to 5.",
        required=False,
    )
    parser.add_argument(
        "--work-dir",
        help="Path where TensorBoard events and checkpoints save.",
        default="output/",
        required=False,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to load checkpoint from, either for evaluation or fine-tuning.",
        required=False,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    cfg = get_config()
    args = parse_args()
    assert (
        args.model_name in model_dict
    ), f"Method {args.model_name} not yet implemented."

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    jax_process = str(f"JAX Process: {jax.process_index()} / {jax.process_count()}")
    jax_devices = str(f"JAX Local Devices: {jax.local_devices()}")
    print(colored(jax_process, "magenta"))
    print(colored(jax_devices, "magenta"))

    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )

    train_ds, test_ds, info = get_jnp_dataset(
        name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        img_shape=[cfg.data_shape[0], cfg.data_shape[1]],
        split_keys=cfg.split_keys,
    )

    steps_per_train = info.splits["train"].num_examples // cfg.batch_size
    steps_per_test = info.splits["test"].num_examples // cfg.batch_size

    learning_rate = cfg.learning_rate

    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)

    if not args.eval_only:
        state = create_train_state(
            model_dict[args.model_name],
            init_rng,
            learning_rate,
            int(cfg.num_classes),
            (
                1,
                cfg.data_shape[0],
                cfg.data_shape[1],
                cfg.data_shape[2],
            ),
        )

        if args.checkpoint_dir:
            assert osp.exists(
                args.checkpoint_dir
            ), f"Checkpoint directory does not exist. Recheck input arguments."
            state = restore_checkpoint(state, checkpoint_dir=args.checkpoint_dir)

        train_and_evaluate(
            state,
            cfg.num_epochs,
            args.work_dir,
            train_ds,
            steps_per_train,
            test_ds,
            steps_per_test,
            cfg.num_classes,
            rng,
        )

    else:
        assert (
            args.checkpoint_dir
        ), f"Checkpoint directory must be specified if evaluating."
        assert osp.exists(
            args.checkpoint_dir
        ), f"Checkpoint directory does not exist. Recheck input arguments."

        state = create_train_state(
            model_dict[args.model_name],
            init_rng,
            learning_rate,
            int(cfg.num_classes),
            (
                1,
                cfg.data_shape[0],
                cfg.data_shape[1],
                cfg.data_shape[2],
            ),
        )

        state = restore_checkpoint(state, checkpoint_dir=args.checkpoint_dir)

        named_tuple = time.localtime()
        time_string = time.strftime("%H:%M:%S", named_tuple)

        test_loss, test_accuracy = test_epoch(
            state, test_ds, int(cfg.num_classes), steps_per_test
        )

        print(f"{' '*10} Accuracy on Test Set: %.2f" % (test_accuracy * 100))
