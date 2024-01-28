from typing import Dict

import jax
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten
import jax.numpy as jnp


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


def calibration_curve(
    outputs: jnp.array, labels: jnp.array, num_bins: int = 20
) -> Dict[str, jnp.array]:
    N = len(labels)
    confidences = jnp.max(outputs, axis=-1)
    preds = jnp.argmax(outputs, axis=-1)

    step = (num_inputs + num_bins - 1) // num_bins
    bins = jnp.sort(confidences)[
        ::step
    ]  # subsamples every `step` element of the array (starts at 0th element)

    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
