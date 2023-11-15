import jax
import jax.numpy as jnp
import numpy as np
import weight_uncertainty.meanfield_vi as meanfield_vi

from jax import jit, vmap
import jax.scipy.stats as stats
import optax
import plotext as plt


def loglikelihood_fn(params, batch):
    logpdf = stats.norm.logpdf(batch, params, 1)
    return jnp.sum(logpdf)


def prior_fn(params):
    return stats.norm.logpdf(params[0], 10, 2)


@jit
def logjoint_fn(params, batch):
    def logjoint(params, batch):
        return prior_fn(params) + loglikelihood_fn(params, batch)

    return vmap(logjoint, in_axes=[0, None])(params, batch)


def data_stream(seed, data, batch_size, data_size):
    """Return an iterator over batches of data."""
    rng = np.random.RandomState(seed)
    num_batches = int(jnp.ceil(data_size / batch_size))
    while True:
        perm = rng.permutation(data_size)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield data[batch_idx]


if __name__ == "__main__":
    optimizer = optax.sgd(1e-5)
    meanfield_vi = meanfield_vi.meanfield_vi(logjoint_fn, optimizer, 30)

    key = jax.random.PRNGKey(123)
    pos = jnp.array([1.0])
    mfvi_state = meanfield_vi.init(pos)

    key, subkey = jax.random.split(key)
    n = 500
    data = jax.random.normal(jax.random.PRNGKey(1), shape=(n, 1)) + 10 * 2
    seed = 1
    batch_size = 50
    batches = data_stream(seed, data, batch_size, n)

    for i in range(500):
        batch = next(batches)
        mfvi_state, mfvi_info, key = meanfield_vi.step(key, mfvi_state, batch)

        if i % 25 == 0:
            print(
                f"Elbo at step {i} | {mfvi_info.elbo} | Log variational: {mfvi_info.log_variational} | Log joint: {mfvi_info.log_joint}"
            )

    samples, key = meanfield_vi.sample(key, mfvi_state, 100)

    plt.hist(list(np.asarray(samples.squeeze())), bins=10)
    plt.title("Posterior over mu (= 20).")
    plt.show()
