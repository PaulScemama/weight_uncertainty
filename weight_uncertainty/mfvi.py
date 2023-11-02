from typing import NamedTuple, Tuple, Callable

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from optax import GradientTransformation, OptState
from jax.random import PRNGKey
from functools import partial


from weight_uncertainty.types import ArrayLikeTree, ArrayTree
from weight_uncertainty.tree_utils import normal_like_tree


# // Structures ----------------------------------------------------------
class MFVIState(NamedTuple):
    mu: ArrayLikeTree
    rho: ArrayLikeTree
    opt_state: OptState


class MFVIInfo(NamedTuple):
    elbo: float


def init(
    position: ArrayLikeTree,
    optimizer: GradientTransformation,
) -> MFVIState:
    """Initialize the mean-field VI state"""
    mu = jax.tree_map(jnp.zeros_like, position)
    rho = jax.tree_map(lambda x: -2.0 * jnp.ones_like(x), position)
    opt_state = optimizer.init((mu, rho))
    return MFVIState(mu, rho, opt_state)


# // Core functions ------------------------------------------------------
def meanfield_logprob(
    meanfield_params: ArrayLikeTree, position: ArrayLikeTree
) -> float:
    """Compute the log probability of parameters contained in `position` under the mean-field
    variational distribution (Diagonal Multivariate Gaussian) governed by mean-field variational
    parameters contained in `meanfield_params`.

    Parameters
    ----------
    meanfield_params : ArrayLikeTree
        Values for parameters governing the variational distribution.
    position : Dict
        Values for data to be evaluated under the variational distribution.

    Returns
    -------
    float
        Log probability of position under the variational distribution.

    Example
    -------
    mu = jnp.zeros((5, 2))
    rho = jnp.ones((5, 2))
    sample = jnp.ones((5, 2))
    params = mu, rho
    print(meanfield_logprob(params, sample))
    """
    mu_tree, rho_tree = meanfield_params
    sigma_tree = jax.tree_map(jnp.exp, rho_tree)

    def meanfield_logprob(position):
        logprob_tree = jax.tree_map(stats.norm.logpdf, position, mu_tree, sigma_tree)
        logprob_tree_flattened, _ = jax.tree_util.tree_flatten(logprob_tree)
        return sum([jnp.sum(leaf) for leaf in logprob_tree_flattened])

    return meanfield_logprob(position)


def meanfield_sample(
    key: PRNGKey, meanfield_params: ArrayLikeTree, n_samples: int
) -> Tuple[ArrayTree, PRNGKey]:
    """
    Sample from variational distribution governed by `meanfield_params`.

    Parameters
    ----------
    meanfield_params : ArrayLikeTree
        Values for parameters governing the variational distribution from which to
        sample from.
    key : PRNGKey
        Key for JAX's pseudo-random number generator.
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    Tuple[ArrayTree, PRNGKey]
        Samples (possibly multiple) drawn from the variational distribution, and
        a fresh pseudo-random number generator.


    Example
    -------
    mu = jnp.zeros((5, 2))
    rho = jnp.ones((5, 2))
    sample = jnp.ones((5, 2))
    params = mu, rho
    print(meanfield_sample(params, jax.random.PRNGKey(1), 2))
    """
    keys = jax.random.split(
        key, n_samples + 1
    )  # n_sample keys for sampling, one extra to return.

    def meanfield_sample_once(key: PRNGKey, meanfield_params: ArrayLikeTree):
        """Sample from variational distribution once."""
        mu_tree, rho_tree = meanfield_params
        sigma_tree = jax.tree_map(jnp.exp, rho_tree)
        noise_tree, new_key = normal_like_tree(mu_tree, key)
        sample = jax.tree_map(
            lambda mu, sigma, noise: mu + sigma * noise,
            mu_tree,
            sigma_tree,
            noise_tree,
        )
        return sample, new_key

    # vmap across keys
    sampled_params, new_keys = jax.vmap(meanfield_sample_once, in_axes=[0, None])(
        keys[1:], meanfield_params
    )
    return sampled_params, new_keys[0]


def meanfield_elbo(
    key: PRNGKey,
    meanfield_params: ArrayLikeTree,
    batch: Tuple[jax.Array],
    logjoint_fn: Callable,
    n_samples: int,
) -> Tuple[float, PRNGKey]:
    """Compute the evidence lower bound for variational parameters and a batch
    of data using the model encompassed in the log joint probability function.
    Additionally compute the gradient of the evidence lower bound with respect
    to the variational parameters.

    Parameters
    ----------
    key : PRNGKey
        Key for JAX's pseudo-random number generator.
    meanfield_params : ArrayLikeTree
        Values for variational parameters governing the variational distribution.
    batch : Tuple
        A batch of data.
    logjoint_fn : _type_
        Function mapping data and parameter values to the log probability
        of the joint distribution which represents the probabilistic model.
    n_samples : int
        Number of samples to draw from the variational distribution during the
        computation of the evidence lower bound.

    Returns
    -------
    Tuple[float, PRNGKey]
        The value for the elbo and a fresh pseudo-random number generator.
    """

    def elbo(meanfield_params):
        sampled_params, new_key = meanfield_sample(key, meanfield_params, n_samples)
        log_variational = meanfield_logprob(meanfield_params, sampled_params)
        log_joint = logjoint_fn(sampled_params, batch).squeeze()
        return (log_variational - log_joint), new_key

    (elbo_value, new_key), elbo_grad = jax.value_and_grad(elbo, has_aux=True)(
        meanfield_params
    )
    return (elbo_value, new_key), elbo_grad


@partial(jax.jit, static_argnames=["logjoint_fn", "optimizer", "n_samples"])
def step(
    key: PRNGKey,
    mfvi_state: MFVIState,
    batch: jax.Array,
    logjoint_fn: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
) -> Tuple[MFVIState, MFVIInfo, PRNGKey]:
    """High-level implementation of mean-field variational inference update step. Computes the
    gradient of the elbo and updates the variational parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        _description_
    mfvi_state : MFVIState
        _description_
    logjoint_fn : callable
        _description_
    optimizer : _type_
        _description_
    batch : _type_
        _description_
    n_samples : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    meanfield_params = mfvi_state.mu, mfvi_state.rho
    # evaluate elbo and get grad
    (elbo, key), grad = meanfield_elbo(
        key, meanfield_params, batch, logjoint_fn, n_samples
    )
    updates, new_opt_state = optimizer.update(
        grad, mfvi_state.opt_state, meanfield_params
    )
    new_mu, new_rho = jax.tree_map(lambda p, u: p + u, meanfield_params, updates)
    new_mfvi_state = MFVIState(new_mu, new_rho, new_opt_state)
    return new_mfvi_state, MFVIInfo(elbo), key


class meanfield_vi:
    init = staticmethod(init)
    step = staticmethod(step)
    sample = staticmethod(meanfield_sample)

    def __new__(
        cls,
        logjoint_fn: Callable,
        optimizer: GradientTransformation,
    ):
        def init_fn(position: ArrayLikeTree):
            return cls.init(position, optimizer)

        def step_fn(
            key: PRNGKey, mfvi_state: MFVIState, batch: jax.Array, n_samples: int
        ):
            return cls.step(key, mfvi_state, batch, logjoint_fn, optimizer, n_samples)

        def sample_fn(
            key: PRNGKey,
            mfvi_state: MFVIState,
            n_samples: int,
        ):
            meanfield_params = mfvi_state.mu, mfvi_state.rho
            return cls.sample(key, meanfield_params, n_samples)

        return MeanfieldVI(init_fn, step_fn, sample_fn)


class MeanfieldVI(NamedTuple):
    init: Callable
    step: Callable
    sample: Callable
