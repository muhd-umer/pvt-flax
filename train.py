from functools import partial
from typing import Union, Iterable
import argparse
import os

"""
Use TPU if avilable
"""
if "TPU_NAME" in os.environ:
    import requests

    if "TPU_DRIVER_MODE" not in globals():
        url = (
            "http:"
            + os.environ["TPU_NAME"].split(":")[1]
            + ":8475/requestversion/tpu_driver_nightly"
        )
        resp = requests.post(url)
        TPU_DRIVER_MODE = 1

    from jax.config import config

    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = os.environ["TPU_NAME"]
    print("Registered TPU:", config.FLAGS.jax_backend_target)
else:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import os.path as osp
import numpy as np
from utils import save_checkpoint, restore_checkpoint
from termcolor import colored
import ml_collections
import time

import jax
from jax import random
import jax.numpy as jnp
import optax
from flax.metrics import tensorboard
from flax.training import train_state
from flax.core import freeze, unfreeze
from flax import jax_utils
import tensorflow as tf
from clu import platform

from data import create_iterator
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


def learning_rate_schedule(cfg: ml_collections.ConfigDict, base_lr: float, steps: int):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=cfg.warmup_epochs * steps,
    )
    cosine_epochs = max(cfg.num_epochs - cfg.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=cosine_epochs * steps
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[cfg.warmup_epochs * steps],
    )
    return schedule_fn


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
        default="",
        help="Path to load checkpoint from, either for evaluation or fine-tuning.",
        required=False,
    )
    args = parser.parse_args()

    return args


def main():
    cfg = get_config()
    args = parse_args()
    assert args.model_name in model_dict, colored(
        f"Method {args.model_name} not yet implemented.", "red"
    )

    if args.checkpoint_dir or args.eval_only:
        assert osp.exists(args.checkpoint_dir), colored(
            f"Checkpoint directory does not exist. Recheck input arguments.", "red"
        )

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

    train_it, train_examples = create_iterator(
        cfg.dataset_name, cfg.batch_size, cfg.data_shape, "train"
    )
    test_it, test_examples = create_iterator(
        cfg.dataset_name, cfg.batch_size, cfg.data_shape, "test"
    )

    train_steps = train_examples // cfg.batch_size
    test_steps = test_examples // cfg.batch_size

    model = model_dict[args.model_name](num_classes=cfg.num_classes)

    base_lr = cfg.learning_rate * cfg.batch_size / 256.0
    schedule = learning_rate_schedule(cfg, base_lr, train_steps)

    def create_train_state(
        rng,
        image_shape: Iterable[int],
    ):
        """
        Creates initial `TrainState`. For more information
            refer to the official Flax documentation.
        https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state
        """

        params = model.init(rng, jnp.ones(image_shape), False)["params"]

        tx = optax.adamw(learning_rate=schedule)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )

        return state

    def train_step(state, inputs, labels, dropout_rng):
        """Perform a single training step."""

        dropout_rng = random.fold_in(dropout_rng, state.step)

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                inputs,
                trainable=True,
                rngs={"dropout": dropout_rng},
            )
            one_hot = jax.nn.one_hot(labels, cfg.num_classes)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        probs = jax.lax.pmean(jax.nn.softmax(logits), axis_name="batch")
        accuracy = jnp.mean(jnp.argmax(probs, -1) == labels)
        grads = jax.lax.pmean(grads, axis_name="batch")

        updated_state = state.apply_gradients(grads=grads)

        return updated_state, loss, accuracy

    def test_step(state, inputs, labels):
        """Perform a single test step."""

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                inputs,
                trainable=False,
            )
            one_hot = jax.nn.one_hot(labels, cfg.num_classes)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), _ = grad_fn(state.params)
        probs = jax.lax.pmean(jax.nn.softmax(logits), axis_name="batch")
        accuracy = jnp.mean(jnp.argmax(probs, -1) == labels)

        return loss, accuracy

    p_train_step = jax.pmap(train_step, axis_name="batch")
    p_test_step = jax.pmap(test_step, axis_name="batch")

    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    dropout_rng = random.split(rng, jax.local_device_count())

    state = create_train_state(
        init_rng,
        (
            1,
            cfg.data_shape[0],
            cfg.data_shape[1],
            3,
        ),
    )
    del init_rng
    if args.checkpoint_dir:
        state = restore_checkpoint(state, checkpoint_dir=args.checkpoint_dir)
    state = jax_utils.replicate(state)

    if not args.eval_only:
        os.makedirs(osp.join(args.work_dir, "logs"), exist_ok=True)
        summary_writer = tensorboard.SummaryWriter(osp.join(args.work_dir, "logs"))
        """
        Main loop for training and evaluation epochs
        """
        for epoch in range(1, cfg.num_epochs + 1):

            named_tuple = time.localtime()
            time_string = time.strftime("%H:%M:%S", named_tuple)

            print(colored(f"[{time_string}] Epoch: {epoch}", "cyan"))

            train_loss, train_accuracy = list(), list()
            test_loss, test_accuracy = list(), list()

            """
            Train epoch
            """
            for _, batch in zip(
                range(train_steps),
                tqdm(
                    train_it,
                    total=train_steps,
                    desc=colored(f"{' '*10} Training", "magenta"),
                    colour="cyan",
                ),
            ):
                inputs, labels = batch["image"], batch["label"]

                state, loss, accuracy = p_train_step(state, inputs, labels, dropout_rng)
                train_loss.append(jax_utils.unreplicate(loss))
                train_accuracy.append(jax_utils.unreplicate(accuracy))

            """
            Test epoch
            """
            for _, batch in zip(
                range(test_steps),
                tqdm(
                    test_it,
                    total=test_steps,
                    desc=colored(f"{' '*10} Training", "magenta"),
                    colour="cyan",
                ),
            ):
                inputs, labels = batch["image"], batch["label"]

                loss, accuracy = p_test_step(state, inputs, labels)
                test_loss.append(jax_utils.unreplicate(loss))
                test_accuracy.append(jax_utils.unreplicate(accuracy))

            train_loss = np.mean(train_loss)
            train_accuracy = np.mean(train_accuracy)
            test_loss = np.mean(test_loss)
            test_accuracy = np.mean(test_accuracy)

            print(
                f"{' '*10} train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
                % (train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
            )
            save_checkpoint(target=state, epoch=epoch, output_dir=args.work_dir)

            """
            Writes tensorboard summary
            """
            summary_writer.scalar("Train_Loss", train_loss, epoch)
            summary_writer.scalar("Train_Accuracy", train_accuracy, epoch)
            summary_writer.scalar("Test_Loss", test_loss, epoch)
            summary_writer.scalar("Test_Accuracy", test_accuracy, epoch)

    else:
        """
        Test epoch
        """
        test_loss, test_accuracy = list(), list()

        for _, batch in zip(
            range(test_steps),
            tqdm(
                test_it,
                total=test_steps,
                desc=colored(f"{' '*10} Training", "magenta"),
                colour="cyan",
            ),
        ):
            inputs, labels = batch["image"], batch["label"]

            loss, accuracy = p_test_step(state, inputs, labels)
            test_loss.append(jax_utils.unreplicate(loss))
            test_accuracy.append(jax_utils.unreplicate(accuracy))

        test_loss = np.mean(test_loss)
        test_accuracy = np.mean(test_accuracy)

        print(f"{' '*10} Accuracy on Test Set: %.2f" % (test_accuracy * 100))


if __name__ == "__main__":
    main()
