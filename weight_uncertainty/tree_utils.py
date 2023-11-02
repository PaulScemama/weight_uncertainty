import jax
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten


def normal_like_tree(a, key):
    treedef = tree_structure(a)
    num_vars = len(tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_map(
        lambda p, k: jax.random.normal(k, shape=p.shape),
        a,
        tree_unflatten(treedef, all_keys[1:]),
    )
    return noise, all_keys[0]
