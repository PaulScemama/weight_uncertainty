import jax
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten
import jax.numpy as jnp


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


def tree_dot(a, b):
    return sum(
        [
            jnp.sum(e1 * e2)
            for e1, e2 in zip(tree_leaves(a), jax.tree_util.tree_leaves(b))
        ]
    )
