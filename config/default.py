import ml_collections


def get_config():
    """
    Configure hyperparameters for training here.
    Data from ConfigDict can be accessed from the 
    outside as any DictLike object.
    Example:
        >>> cfg = get_config()
        >>> print(cfg.learning_rate)
        >>> 0.1
    """
    config = ml_collections.ConfigDict()

    config.learning_rate = 3.5e-4
    config.batch_size = 128
    config.num_epochs = 10

    return config
