import jax

import optax

from datasets import load_dataset
import numpy as np


mnist_data = load_dataset("mnist")
data_train, data_test = mnist_data["train"], mnist_data["test"]

X_train = np.stack([np.array(example["image"]) for example in data_train])
y_train = np.array([example["label"] for example in data_train])

X_test = np.stack([np.array(example["image"]) for example in data_test])
y_test = np.array([example["label"] for example in data_test])

import jax.numpy as jnp


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


X_train, y_train, N_train = prepare_data(X_train, y_train)
X_test, y_test, N_test = prepare_data(X_test, y_test)

import flax.linen as nn
import jax.scipy.stats as stats


class NN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=500)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return nn.log_softmax(x)


model = NN()


def logprior_fn(params):
    """Compute the value of the log-prior density function."""
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(stats.norm.logpdf(flat_params))


def loglikelihood_fn(params, data):
    """Categorical log-likelihood"""
    X, y = data
    return jnp.sum(y * model.apply(params, X))


loglikelihood_fn = jax.vmap(loglikelihood_fn, in_axes=[0, None])


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


import weight_uncertainty.meanfield_vi as mfvi


for lr in [1e-3]:
    optimizer = optax.sgd(lr)
    meanfield_vi = mfvi.meanfield_vi(
        loglikelihood_fn,
        optimizer,
        30,
        logprior_name="unit_gaussian",
    )

    key = jax.random.PRNGKey(123)
    key, subkey = jax.random.split(key)
    pos = model.init(subkey, jnp.ones(X_train.shape[-1]))
    mfvi_state = meanfield_vi.init(pos)

    key, subkey = jax.random.split(key)
    n = N_train.item()
    seed = 1
    batch_size = 256
    batches = data_stream(seed, (X_train, y_train), batch_size, n)

    # Sample from the posterior
    accuracies = []
    steps = []

    print(f"LEARNING RATE: {lr}")
    for i in range(5_000):
        batch = next(batches)
        mfvi_state, mfvi_info, key = meanfield_vi.step(key, mfvi_state, batch)

        if i % 100 == 0:
            output = jax.jit(compute_predictions)(mfvi_state.mu, X_test)
            acc = compute_accuracy(output, y_test)
            print(
                f"Elbo at step {i} | {mfvi_info.elbo} | Log variational: {mfvi_info.log_variational} | Log joint: {mfvi_info.log_joint} | Acc at step {i} | {acc}"
            )
