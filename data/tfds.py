from typing import Iterable
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf


def transform_images(row, shape):
    x_train = tf.image.resize_with_pad(row["image"], shape[0], shape[1])

    return {"image": x_train, "label": row["label"]}


def get_jnp_dataset(
    name: str, batch_size: int, img_shape: Iterable[int], split_keys: Iterable[str]
):
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
    train_data, info = tfds.load(name=name, split=split_keys[0], with_info=True)
    train_data = train_data.map(lambda row: transform_images(row, img_shape))
    train_data = train_data.repeat().cache().batch(batch_size)

    test_data = tfds.load(name=name, split=split_keys[1])
    test_data = test_data.map(lambda row: transform_images(row, img_shape))
    test_data = test_data.repeat().cache().batch(batch_size)

    train_ds = tfds.as_numpy(train_data)
    test_ds = tfds.as_numpy(test_data)

    return train_ds, test_ds, info
