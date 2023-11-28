import jax
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten


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


# Useful PyTree Utility: modified from https://github.com/google-research/google-research/blob/master/bnn_hmc/utils/tree_utils.py
# to allow for `n_samples`` to be taken.
def normal_like_tree(a, key, n_samples):
    treedef = tree_structure(a)
    num_vars = len(tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_map(
        lambda p, k: jax.random.normal(k, shape=(n_samples,) + p.shape),
        a,
        tree_unflatten(treedef, all_keys[1:]),
    )
    return noise, all_keys[0]
