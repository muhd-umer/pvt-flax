from sched import scheduler
from typing import Union, Iterable, Any
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import os.path as osp
from unittest import runner
import numpy as np
import ml_collections
from utils import save_checkpoint
from termcolor import colored
import time

import jax
from jax import random
import jax.numpy as jnp
import optax
from flax.metrics import tensorboard
from flax.training import train_state
import tensorflow as tf
from clu import platform

from data import get_jnp_dataset
from tqdm import tqdm
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

model_dict = {
    "PVT_V2_B0": PVT_V2_B0,
    "PVT_V2_B1": PVT_V2_B1,
    "PVT_V2_B2": PVT_V2_B2,
    "PVT_V2_B3": PVT_V2_B3,
    "PVT_V2_B4": PVT_V2_B4,
    "PVT_V2_B5": PVT_V2_B5,
}


def learning_rate_schedule(
    warmup_epochs, num_epochs, base_learning_rate, steps_per_epoch
):
    """
    Create learning rate schedule.
    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


@jax.pmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def step(state, inputs, labels, num_classes, trainable, rng):
    """
    Defines a single step (forward pass).
    Returns:
        if trainable:
            state: Updated state of the model
        loss: Mean loss for the current batch
        accuracy: Mean accuracy for the current batch
    """

    def loss_fn(params, num_classes, trainable, rng):
        logits = state.apply_fn(
            {"params": params}, inputs, trainable=trainable, rngs={"dropout": rng}
        )
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

        return loss, logits

    if trainable:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params, num_classes, trainable, rng)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        state = update_model(state, grads)

        return state, loss, accuracy

    loss, logits = loss_fn(state.params, num_classes, trainable, rng)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    return loss, accuracy


step = jax.pmap(step, axis_name="ensemble", static_broadcasted_argnums=(3, 4))


def create_train_state(
    params,
    model,
    warmup_epochs,
    num_epochs,
    base_lr,
    steps,
):
    """
    Creates initial `TrainState`. For more information
        refer to the official Flax documentation.
    https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state
    """

    schedule = learning_rate_schedule(warmup_epochs, num_epochs, base_lr, steps)
    tx = optax.adamw(learning_rate=schedule)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    return state


create_train_state = jax.pmap(
    create_train_state, static_broadcasted_argnums=tuple(range((1, 6))
)


def train_and_evaluate(
    state: train_state.TrainState,
    epochs: int,
    work_dir: Union[os.PathLike, str],
    train_ds,
    test_ds,
    num_classes,
    info,
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

    for epoch in range(1, epochs + 1):
        rng, init_rng = random.split(rng)

        train_loss, train_accuracy = list(), list()
        test_loss, test_accuracy = list(), list()
        total_train = info.splits["train"].num_examples
        total_test = info.splits["test"].num_examples
        named_tuple = time.localtime()
        time_string = time.strftime("%H:%M:%S", named_tuple)

        print(colored(f"[{time_string}] Epoch: {epoch}", "cyan"))

        for batch in tqdm(
            train_ds,
            total=total_train,
            desc=colored(f"{' '*10} Training", "magenta"),
            colour="cyan",
        ):
            inputs, labels = batch["image"], batch["label"]
            inputs = jax.jax_utils.replicate(jnp.float32(inputs) / 255.0)
            labels = jax.jax_utils.replicate(jnp.float32(labels))

            state, train_loss_batch, train_accuracy_batch = step(
                state, inputs, labels, int(num_classes), True, init_rng
            )
            train_loss.append(jax.jax_utils.unreplicate(train_loss_batch))
            train_accuracy.append(jax.jax_utils.unreplicate(train_accuracy_batch))

        for batch in tqdm(
            test_ds,
            total=total_test,
            desc=colored(f"[{time_string}] Validating", "magenta"),
            colour="cyan",
        ):
            inputs, labels = batch["image"], batch["label"]
            inputs = jax.jax_utils.replicate(jnp.float32(inputs) / 255.0)
            labels = jax.jax_utils.replicate(jnp.float32(labels))

            test_loss_batch, test_accuracy_batch = step(
                state, inputs, labels, int(num_classes), False, init_rng
            )
            test_loss.append(jax.jax_utils.unreplicate(test_loss_batch))
            test_accuracy.append(jax.jax_utils.unreplicate(test_accuracy_batch))

        train_loss = sum(train_loss) / len(train_loss)
        train_accuracy = sum(train_accuracy) / len(train_accuracy)
        test_loss = sum(test_loss) / len(test_loss)
        test_accuracy = sum(test_accuracy) / len(test_accuracy)

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

    steps_per_epoch = info.splits["train"].num_examples
    if cfg.num_train_steps == -1:
        num_steps = int(steps_per_epoch * cfg.num_epochs)
    else:
        num_steps = cfg.num_train_steps

    if cfg.steps_per_eval == -1:
        num_validation_examples = info.splits["test"].num_examples
        steps_per_eval = num_validation_examples // cfg.batch_size
    else:
        steps_per_eval = cfg.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 10
    base_learning_rate = cfg.learning_rate * cfg.batch_size / 256

    model, params = create_PVT_V2(
        model_dict[args.model_name],
        rng=random.PRNGKey(0),
        num_classes=cfg.num_classes,
        in_shape=(
            1,
            cfg.data_shape[0],
            cfg.data_shape[1],
            cfg.data_shape[2],
        ),
        checkpoint=args.checkpoint_dir,
        distributed=True
    )

    lr_list = [cfg.warmup_epochs, cfg.num_epochs, base_learning_rate, steps_per_epoch]

    if not args.eval_only:
        state = create_train_state(
            model=model,
            params=params,
            warmup_epochs=cfg.warmup_epochs,
            num_epochs=cfg.num_epochs,
            base_lr=base_learning_rate,
            steps=steps_per_epoch,
        )

        train_and_evaluate(
            state=state,
            epochs=cfg.num_epochs,
            work_dir=args.work_dir,
            train_ds=train_ds,
            test_ds=test_ds,
            num_classes=cfg.num_classes,
            info=info,
        )

    else:
        assert (
            args.checkpoint_dir
        ), f"Checkpoint directory must be specified if evaluating."
        rng, init_rng = random.split(random.PRNGKey(0))

        state = create_train_state(
            model=model,
            params=params,
            warmup_epochs=cfg.warmup_epochs,
            num_epochs=cfg.num_epochs,
            base_lr=base_learning_rate,
            steps=steps_per_epoch,
        )

        named_tuple = time.localtime()
        time_string = time.strftime("%H:%M:%S", named_tuple)
        test_loss, test_accuracy = list(), list()
        total_test = info.splits["test"].num_examples

        for batch in tqdm(
            test_ds,
            total=total_test,
            desc=colored(f"[{time_string}] Evaluation", "cyan"),
            colour="cyan",
        ):
            test_loss_batch, test_accuracy_batch = step(
                state, batch, int(cfg.num_classes), False, init_rng
            )
            test_loss.append(test_loss_batch)
            test_accuracy.append(test_accuracy_batch)

        test_loss = sum(test_loss) / len(test_loss)
        test_accuracy = sum(test_accuracy) / len(test_accuracy)
        print(f"{' '*10} Accuracy on Test Set: %.2f" % (test_accuracy * 100))