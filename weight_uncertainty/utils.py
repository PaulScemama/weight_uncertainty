import jax


def print_visible_devices():
    print("JAX sees the following devices:", jax.devices())
    try:
        import tensorflow as tf

        print("TF sees the following devices:", tf.config.get_visible_devices())
    except ImportError:
        print(f"Tensorflow is not installed.")


def hide_gpu_from_tf():
    try:
        import tensorflow as tf

        tf.config.experimental.set_visible_devices([], "GPU")
    except ImportError:
        raise ImportError("Need to install tensorflow to use this function.")
