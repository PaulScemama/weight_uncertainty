import jax
import jax.numpy as jnp
import numpy as np
from weight_uncertainty.meanfield_vi import MeanFieldVI
from datasets import Dataset
import jax.scipy.stats as stats
import optax
import plotext as plt
from jax.random import split


def gen_data(key, n):
    data = jax.random.normal(key, shape=(n, 1)) + 10 * 2
    return data


def loglikelihood_fn(params, batch):
    logpdf = stats.norm.logpdf(batch["y"], params, 1)
    return jnp.sum(logpdf)


def main():

    # Prepare data
    key = jax.random.PRNGKey(123)
    key, data_key = split(key)
    n = 250
    data = gen_data(data_key, n)
    batch_size = 50
    batches = Dataset.from_dict({"y": data}).with_format("jax").iter(batch_size)

    # Create MeanFieldVI inference engine
    optimizer = optax.sgd(1e-3)
    n_samples = 15
    init, step, sample_params = MeanFieldVI(
        loglikelihood_fn=loglikelihood_fn, optimizer=optimizer, n_samples=n_samples
    )

    # Run Meanfield VI
    num_steps = 50
    eval_every = 5
    state = init(params=jnp.array([1.0]))

    key, *training_keys = split(key, num_steps + 1)
    for i, rng_key in enumerate(training_keys):
        batch = next(batches)
        state, info = step(rng_key, state, batch)
        if i % eval_every == 0:
            print(f"Step {i} | elbo: {info.elbo} | nll: {info.nll} | kl: {info.kl}")

    # Generate posterior over parameters
    key, sampling_key = split(key)
    samples = sample_params(
        sampling_key,
        state,
    )
    plt.hist(list(np.asarray(samples.squeeze())), bins=10)
    plt.title("Posterior over mu (= 20).")
    plt.show()


if __name__ == "__main__":
    main()
