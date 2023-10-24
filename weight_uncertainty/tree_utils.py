import jax


def normal_like_tree(a, key):
    treedef = jax.tree_structure(a)
    num_vars = len(jax.tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_map(
        lambda p, k: jax.random.normal(k, shape=p.shape),
        a,
        jax.tree_unflatten(treedef, all_keys[1:]),
    )
    return noise, all_keys[0]
