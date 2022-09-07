import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf


def get_jnp_dataset(name: str, batch_size: int):
    """
    Load "name" train and test data into memory;
    General Feature Structure:
        FeaturesDict({
            'image': Image(shape=(None, None, 3), dtype=tf.uint8),
            'image/filename': Text(shape=(), dtype=tf.string),
            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=2),
        })
    Note: This feature structure varies from dataset to dataset.
    For more information, refer to:
        https://www.tensorflow.org/datasets/catalog/overview
    Returns:
        Train and Test data with features_dict.
    """
    ds_builder = tfds.builder(name=name)
    ds_builder.download_and_prepare()

    train_ds = ds_builder.as_dataset(split="train", batch_size=batch_size)
    test_ds = ds_builder.as_dataset(split="test", batch_size=batch_size)

    train_ds = tfds.as_numpy(train_ds)
    test_ds = tfds.as_numpy(test_ds)

    # train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
    # test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0

    return train_ds, test_ds
