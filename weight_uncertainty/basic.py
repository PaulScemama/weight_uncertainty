import jax
import jax.random as random
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import Array
from functools import partial
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker

"""
This file contains functions necessary for constructing a Bayesian Neural Network, 
including the loss function it uses to optimize the parameters theta = (mu, rho). 

Functions
---------
sigmas_from_rhos(rhos)

        3.2 | "We parameterise the standard deviation pointwise as σ = log(1 + exp(ρ))"

            
logvariational_fn(weights, mus, rhos)

        3.1 | "F(D, θ) ≈ log q(w|θ) − log P(w) − log P(D|w)"                         
                         ^^^^^^^^^^
        3.2 | "Suppose that the variational posterior is a diagonal Gaussian distribution"
    
        
samplevariational_fn(n_samples, mus, rhos)

        3.2 | "w = t(θ, epsilon) = µ + log(1 + exp(ρ)) ◦ epsilon"
            
logprior_fn(weights, pi, var1, var2)

        3.3 | "P(w) = π N(w |0, σ2_1) + (1 − π) N(w |0, σ2_2)"
"""

# @jax.jit
def rhos_from_sigmas(sigmas):
    """
    For testing.
    """
    return jnp.log(jnp.exp(sigmas) - 1)


def sigmas_from_rhos(rhos: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Computes the standard deviations from the the rho parameters according to the
    parameterization σ = log(1 + exp(ρ)) introduced in section 3.2 of "Weight
    Uncertainty in Neural Networks".

    Parameters
    ----------
    rhos : [...] float tensor
        Rho parameters for ... Gaussian distributions that represent the weights of
        the network.

    Returns
    -------
    sigmas : [...] float tensor
        Sigma parameters for ... Gaussian distributions that represent the weights of
        the network.
    """
    return jnp.log(1 + jnp.exp(rhos))


# @jax.jit
def logvariational_fn(
    weights: Float[Array, "..."],
    mus: Float[Array, "..."],
    rhos: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Computes the log density of each individual weight under a normal distribution
    governed by the corresponding (mu, rho).

    Parameters
    ----------
        weights : [...] float tensor
            weights to be evaluated.
        mus : [...] float tensor
            mus each governing an individual normal distribution which
            the corresponding weight will be evaluated under.
        rhos : [...] float tensor
            rhos each governing an individual normal distribution which
            the corresponding wiehgt will be evaluated under.

    Returns
    -------
        [...] float tensor
            the M log probabilities evaluated for each of the Normal(w|mu,rho)
            distributions.

    Notes
    -----
        We define sigmas = log(1 + exp(rhos)) as in section 3.2 of "Weight Uncertainty
        in Neural Networks".
    """
    sigmas = sigmas_from_rhos(rhos)
    # Unravel weights, mus, sigmas
    weights_unraveled = weights.ravel()
    mus_unraveled = mus.ravel()
    sigmas_unraveled = sigmas.ravel()
    # Evaluate each weight under its corresponding unvariate Gaussian
    return stats.norm.logpdf(weights_unraveled, mus_unraveled, sigmas_unraveled).mean()



def samplevariational_fn(
    mus: Float[Array, "..."],
    rhos: Float[Array, "..."],
    key: random.PRNGKey,
) -> Float[Array, "..."]:
    """
    Samples from M gaussians governed by the M corresponding mus
    and rhos.

    Parameters
    ----------
        mus : [...] float tensor
            mus each governing an individual normal distribution which
            the corresponding weight will be evaluated under.
        rhos : [...] float tensor
            rhos each governing an individual normal distribution which
            the corresponding weight will be evaluated under.
    Returns
    -------
    [n_samples x ...] float tensor
        n_samples of an [...] tensor that represents M samples from M
        independent univariate gaussian distributions.

    Notes
    -----
        We define sigmas = log(1 + exp(rhos)) as in section 3.2 of "Weight Uncertainty
        in Neural Networks"
    """
    shape = mus.shape
    epsilons = random.normal(key, shape)
    sigmas = sigmas_from_rhos(rhos)
    return mus + sigmas * epsilons


# @jax.jit
def logprior_fn(
    weights: Float[Array, "..."], pi: float, var1: float, var2: float
) -> jax.Array:
    """
    Computes the log density of each scalar weight under a scale mixture of
    two zero-mean univariate Gaussians governed by the parameters pi, var1, var2
    (all shared between each weight scalar).

    Parameters
    ----------
    weights : [...] float tensor
        weights representing the weights of the network to be
        evaluated.
    pi : float
        the proportion of scaling between the mixture of the two Gaussians
    var1 : float
        the variance of the first Gaussian
    var2 : float
        the variance of the second Gaussian

    Returns
    -------
    [...] float tensor
        the log density of each scalar weight evaluated under the same univariate
        scale mixture prior distribution governed by the parameters pi, var1, and var2.
    """
    weights_unraveled = weights.ravel()
    # sqrt(var_) since norm.lopdf expects standard deviation
    gaussian1_log_prob = stats.norm.logpdf(weights_unraveled, 0, jnp.sqrt(var1))
    gaussian2_log_prob = stats.norm.logpdf(weights_unraveled, 0, jnp.sqrt(var2))
    return jnp.log(
        pi * jnp.exp(gaussian1_log_prob) + (1 - pi) * jnp.exp(gaussian2_log_prob)
    ).mean()


class ScaleMixturePrior:

    def __init__(self, pi: float, var1: float, var2: float):
        self.pi = pi
        self.var1 = var1
        self.var2 = var2
    
    def __call__(self, weights):
        weights_unraveled = weights.ravel()
        # sqrt(var_) since norm.lopdf expects standard deviation
        gaussian1_log_prob = stats.norm.logpdf(weights_unraveled, 0, jnp.sqrt(self.var1))
        gaussian2_log_prob = stats.norm.logpdf(weights_unraveled, 0, jnp.sqrt(self.var2))
        return jnp.log(
            self.pi * jnp.exp(gaussian1_log_prob) + (1 - self.pi) * jnp.exp(gaussian2_log_prob)
        ).mean()
    

# def logprior_fn2(params):
#     """Computes the Gaussian prior log-density."""
#     # ToDo izmailovpavel: make temperature treatment the same as in gaussian
#     # likelihood function.
#     n_params = sum([p.size for p in jax.tree_leaves(params)])
#     log_prob = -(0.5 * tree_utils.tree_dot(params, params) * weight_decay +
#                  0.5 * n_params * jnp.log((2 * math.pi) / weight_decay))
#     return log_prob / temperature
