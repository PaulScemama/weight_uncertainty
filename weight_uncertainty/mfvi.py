from typing import NamedTuple, Dict

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from jax.random import PRNGKey
from functools import partial

import tree_utils


# // Structures ----------------------------------------------------------
class MFVIState(NamedTuple):
    mu: Dict
    rho: Dict
    opt_state: Dict


class MFVIInfo(NamedTuple):
    elbo: float


def init(
    position: Dict,
    optimizer,
) -> MFVIState:
    """Initialize the mean-field VI state"""
    mu = jax.tree_map(jnp.zeros_like, position)
    rho = jax.tree_map(lambda x: -2.0 * jnp.ones_like(x), position)
    opt_state = optimizer.init((mu, rho))
    return MFVIState(mu, rho, opt_state)


# // Core functions ------------------------------------------------------
def meanfield_logprob(meanfield_params, sample_tree: Dict):
    # compute log probability of `position` under variational distribution
    # governed by `meanfield_params`.
    mu_tree, rho_tree = meanfield_params
    sigma_tree = jax.tree_map(jnp.exp, rho_tree)

    def meanfield_logprob(sample_tree):
        logprob_tree = jax.tree_map(stats.norm.logpdf, sample_tree, mu_tree, sigma_tree)
        logprob_tree_flattened, _ = jax.tree_util.tree_flatten(logprob_tree)
        return sum([jnp.sum(leaf) for leaf in logprob_tree_flattened])

    return meanfield_logprob(sample_tree)


# mu = jnp.zeros((5, 2))
# rho = jnp.ones((5, 2))
# sample = jnp.ones((5, 2))
# params = mu, rho
# meanfield_logprob = mfvi.meanfield_logprob(params, sample)
# print(meanfield_logprob)


def meanfield_sample(meanfield_params, key: PRNGKey, n_samples: int):
    # sample from the variational distribution governed by
    # `meanfield_params`
    keys = jax.random.split(
        key, n_samples + 1
    )  # n_sample keys for sampling, one extra to return.

    def meanfield_sample(meanfield_params, key):
        mu_tree, rho_tree = meanfield_params
        sigma_tree = jax.tree_map(jnp.exp, rho_tree)

        noise_tree, new_key = tree_utils.normal_like_tree(mu_tree, key)
        sample = jax.tree_map(
            lambda mu, sigma, noise: mu + sigma * noise,
            mu_tree,
            sigma_tree,
            noise_tree,
        )

        return sample, new_key

    sampled_params, new_keys = jax.vmap(meanfield_sample, in_axes=[None, 0])(
        meanfield_params, keys[1:]
    )
    return sampled_params, new_keys[0]


# mu = jnp.zeros((5, 2))
# rho = jnp.ones((5, 2))
# sample = jnp.ones((5, 2))
# params = mu, rho
# mfvi.meanfield_sample(params, jax.random.PRNGKey(1), 2)


# @partial(jax.jit, static_argnames=["logjoint_fn, n_samples"])
def meanfield_elbo(meanfield_params, batch, key, logjoint_fn, n_samples):
    # compute the elbo given a `batch` of data, the `logjoint_fn` joint density that
    # is proportional to the target posterior, and the `meanfield_params`.
    sampled_params, new_key = meanfield_sample(meanfield_params, key, n_samples)
    log_variational = meanfield_logprob(meanfield_params, sampled_params)
    log_joint = logjoint_fn(sampled_params, batch).squeeze()
    return (log_variational - log_joint), new_key


# Update function ----------------------------------------------------------------
@partial(jax.jit, static_argnames=["logjoint_fn", "optimizer", "n_samples"])
def step(
    key: jax.random.PRNGKey,
    mfvi_state: MFVIState,
    logjoint_fn: callable,
    optimizer,
    batch,
    n_samples,
):
    meanfield_params = mfvi_state.mu, mfvi_state.rho
    # evaluate elbo and get grad
    (elbo, key), grad = jax.value_and_grad(meanfield_elbo, has_aux=True)(
        meanfield_params, batch, key, logjoint_fn, n_samples
    )

    updates, new_opt_state = optimizer.update(
        grad, mfvi_state.opt_state, meanfield_params
    )
    new_mu, new_rho = jax.tree_map(lambda p, u: p + u, meanfield_params, updates)
    new_mfvi_state = MFVIState(new_mu, new_rho, new_opt_state)
    return new_mfvi_state, MFVIInfo(elbo), key
