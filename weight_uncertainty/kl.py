import jax
import jax.numpy as jnp

from jax.tree_util import tree_leaves


def unit_gaussian_kl(meanfield_params):
    mu, rho = meanfield_params
    sigma = jax.tree_map(jnp.exp, rho)

    def kl(mu, sigma):
        """From AutoEncoding Variational Bayes Appendix B."""
        mu_squared = jnp.square(mu)
        sigma_squared = jnp.square(sigma)
        return -jnp.log(sigma_squared) + (sigma_squared + mu_squared) / 2 - (1 / 2)

    kl_tree = jax.tree_map(kl, mu, sigma)
    kl_val = sum([param_kl.sum() for param_kl in tree_leaves(kl_tree)])
    return kl_val


def isotropic_gaussian_kl(meanfield_params, wd):
    mu, rho = meanfield_params
    sigma = jax.tree_map(jnp.exp, rho)

    def kl(mu, sigma):
        mu_squared = jnp.square(mu)
        sigma_squared = jnp.square(sigma)
        wd_squared = jnp.square(wd)
        return (
            jnp.log(wd / sigma)
            + (sigma_squared + mu_squared) / (2 * wd_squared)
            - (1 / 2)
        )

    kl_tree = jax.tree_map(kl, mu, sigma)
    kl_val = sum([param_kl.sum() for param_kl in tree_leaves(kl_tree)])
    return kl_val
