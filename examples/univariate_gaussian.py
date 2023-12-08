import jax
import jax.numpy as jnp
import numpy as np
from weight_uncertainty.interface import meanfield_vi
from datasets import Dataset
import jax.scipy.stats as stats
import optax
import plotext as plt
from jax.random import split
from jax import lax


def gen_data(key, n):
    data = jax.random.normal(key, shape=(n, 1)) + 10 * 2
    return data


def loglikelihood_fn(params, batch):
    logpdf = stats.norm.logpdf(batch["y"], params, 1)
    return jnp.sum(logpdf)


def run_inference_algorithm(
    rng_key,
    initial_state_or_position,
    inference_algorithm,
    batches,
    num_steps,
) -> tuple:
    """Wrapper to run an inference algorithm.

    Parameters
    ----------
    rng_key : PRNGKey
        The random state used by JAX's random numbers generator.
    initial_state_or_position: ArrayLikeTree
        The initial state OR the initial position of the inference algorithm. If an initial position
        is passed in, the function will automatically convert it into an initial state.
    inference_algorithm : Union[SamplingAlgorithm, VIAlgorithm]
        One of blackjax's sampling algorithms or variational inference algorithms.
    num_steps : int
        Number of learning steps.

    Returns
    -------
    Tuple[State, State, Info]
        1. The final state of the inference algorithm.
        2. The history of states of the inference algorithm.
        3. The history of the info of the inference algorithm.
    """
    try:
        initial_state = inference_algorithm.init(initial_state_or_position)
    except TypeError:
        # We assume initial_state is already in the right format.
        initial_state = initial_state_or_position

    keys = split(rng_key, num_steps)

    @jax.jit
    def one_step(state, rng_key):
        batch = next(batches)
        state, info = inference_algorithm.step(rng_key, state, batch)
        return state, (state, info)

    final_state, (state_history, info_history) = lax.scan(one_step, initial_state, keys)
    return final_state, state_history, info_history


if __name__ == "__main__":
    key = jax.random.PRNGKey(123)
    key, data_key, inference_key = split(key, 3)
    n = 250
    data = gen_data(data_key, n)
    batch_size = 50

    # key must match loglikelihood_fn
    batches = Dataset.from_dict({"y": data}).with_format("jax").iter(batch_size)

    optimizer = optax.sgd(1e-3)
    meanfield_vi = meanfield_vi(
        loglikelihood_fn=loglikelihood_fn,
        logprior="unit_gaussian",
        optimizer=optimizer,
        n_samples=30,
    )

    initial_pos = jnp.array([1.0])
    final_state, state_history, info_history = run_inference_algorithm(
        inference_key, initial_pos, meanfield_vi, batches, 500
    )

    key, sampling_key = split(key)
    samples = meanfield_vi.sample(key, final_state, 100)

    plt.hist(list(np.asarray(samples.squeeze())), bins=10)
    plt.title("Posterior over mu (= 20).")
    plt.show()
