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
    config.warmup_epochs = 2
    config.momentum = 0.9
    
    config.num_epochs = 10
    config.log_every_steps = 100

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    # configure input dataset keys
    config.dataset_name = "cifar10"
    config.data_shape = [32, 32]
    config.num_classes = 10

    return config
