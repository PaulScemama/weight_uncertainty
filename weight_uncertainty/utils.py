import jax
import tensorflow as tf


def print_visible_devices():
    print("JAX sees the following devices:", jax.devices())
    print("TF sees the following devices:", tf.config.get_visible_devices())


def hide_gpu_from_tf():
    tf.config.experimental.set_visible_devices([], "GPU")
