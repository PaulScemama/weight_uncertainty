import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset

import flax.linen as nn
import jax.scipy.stats as stats
from weight_uncertainty.new_interface import meanfield_vi
from jax.random import split
from jax import lax


def one_hot_encode(x, k):
    "Create a one-hot encoding of x of size k."
    return jnp.array(x[:, None] == jnp.arange(k), dtype=jnp.float32)


@jax.jit
def prepare_data(X, y, num_categories=10):
    y = one_hot_encode(y, num_categories)

    num_examples = X.shape[0]
    num_pixels = 28 * 28
    X = X.reshape(num_examples, num_pixels)
    X = X / 255.0

    return X, y, num_examples


def data_stream(seed, data, batch_size, data_size):
    """Return an iterator over batches of data."""
    rng = np.random.RandomState(seed)
    num_batches = int(jnp.ceil(data_size / batch_size))
    while True:
        perm = rng.permutation(data_size)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield data[0][batch_idx], data[1][batch_idx]


class NN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=500)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return nn.log_softmax(x)


def logprior_fn(params):
    """Compute the value of the log-prior density function."""
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(stats.norm.logpdf(flat_params))


def loglikelihood_fn(params, data):
    """Categorical log-likelihood"""
    X, y = data
    return jnp.sum(y * model.apply(params, X))


def compute_predictions(sampled_params, X):
    out = model.apply(sampled_params, X)
    return out


@jax.jit
def compute_accuracy(outputs, y):
    """Compute the accuracy of the model.

    To make predictions we take the number that corresponds to the highest
    probability value, which corresponds to a 1-0 loss.

    """
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(outputs, axis=1)
    return jnp.mean(predicted_class == target_class)


def run_inference_algorithm(
    rng_key,
    initial_state_or_position,
    inference_algorithm,
    batches,
    num_steps,
    eval_every,
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

    state = initial_state
    for i, key in enumerate(keys):
        batch = next(batches)
        state, info = jax.jit(inference_algorithm.step)(key, state, batch)

        if i % eval_every == 0:
            output = jax.jit(compute_predictions)(state.mu, X_test)
            acc = compute_accuracy(output, y_test)
            print(f"Epoch {i} | Acc: {acc} | Elbo: {info.elbo}")

    return state


if __name__ == "__main__":
    # -------------------------------- Data --------------------------------
    mnist_data = load_dataset("mnist")
    data_train, data_test = mnist_data["train"], mnist_data["test"]

    X_train = np.stack([np.array(example["image"]) for example in data_train])
    y_train = np.array([example["label"] for example in data_train])

    X_test = np.stack([np.array(example["image"]) for example in data_test])
    y_test = np.array([example["label"] for example in data_test])

    X_train, y_train, N_train = prepare_data(X_train, y_train)
    X_test, y_test, N_test = prepare_data(X_test, y_test)

    model = NN()
    optimizer = optax.sgd(1e-3)
    meanfield_vi = meanfield_vi.init_w_iso_gauss(loglikelihood_fn, optimizer, 15)

    key = jax.random.PRNGKey(123)
    key, subkey = jax.random.split(key)
    pos = model.init(subkey, jnp.ones(X_train.shape[-1]))
    mfvi_state = meanfield_vi.init(pos)

    key, subkey = jax.random.split(key)
    n = N_train.item()
    seed = 1
    batch_size = 100
    batches = data_stream(seed, (X_train, y_train), batch_size, n)

    final_state = run_inference_algorithm(key, pos, meanfield_vi, batches, 1000, 50)
