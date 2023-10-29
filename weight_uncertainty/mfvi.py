import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from jax.random import PRNGKey


import tree_utils


def meanfield_logprob(meanfield_params, sample_tree):
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

    def meanfield_sample(key, _): # _ for lax.scan's f arg signature.
        mu_tree, rho_tree = meanfield_params
        sigma_tree = jax.tree_map(jnp.exp, rho_tree)

        noise_tree, new_key = tree_utils.normal_like_tree(mu_tree, key)
        sample = jax.tree_map(
            lambda mu, sigma, noise: mu + sigma * noise,
            mu_tree,
            sigma_tree,
            noise_tree,
        )

        return new_key, sample

    new_key, sampled_params = jax.lax.scan(meanfield_sample, init=key, xs=jnp.arange(n_samples))

    return sampled_params, new_key


# mu = jnp.zeros((5, 2))
# rho = jnp.ones((5, 2))
# sample = jnp.ones((5, 2))
# params = mu, rho
# mfvi.meanfield_sample(params, jax.random.PRNGKey(1), 2)


def meanfield_elbo(meanfield_params, batch, key, logjoint_fn, n_samples):
    # compute the elbo given a `batch` of data, the `logjoint_fn` joint density that
    # is proportional to the target posterior, and the `meanfield_params`.
    sampled_params, new_key = meanfield_sample(meanfield_params, key, n_samples)
    log_variational = meanfield_logprob(meanfield_params, sampled_params)
    log_joint = logjoint_fn(sampled_params, batch)
    return (log_variational - log_joint), new_key


def meanfield_approximate(
    key,
    state,
    logjoint_fn: int,
    n_samples: int,
    data_iter,
):
    pass
