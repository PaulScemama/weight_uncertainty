from flax import linen as nn 
from flax.linen.module import compact
from typing import Callable, Dict
import jax.numpy as jnp
from jaxtyping import Float, Array
import jax
import jax.scipy.stats as stats
import jax.random as random


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
    key, subkey = jax.random.split(key)
    shape = mus.shape
    epsilons = random.normal(subkey, shape)
    sigmas = sigmas_from_rhos(rhos)
    return mus + sigmas * epsilons, key



def sample_predictive(
        n_samples: int,
        model: Callable,
        init_state: Dict,
        params: Dict,
        x: jnp.array,
):
    f = lambda state, _: tuple(reversed(model.apply(params, state, x)))
    final_state, (logits_stacked, _, _) = jax.lax.scan(
        f, init=init_state, xs=jnp.arange(n_samples)
    )
    return logits_stacked, final_state


    





class MeanFieldLinearLayer(nn.Module):

    features: int
    logprior: Callable
    parameter_init: Callable = nn.initializers.lecun_normal()

    @compact
    def __call__(self, state: Dict, inputs: jnp.array):
        # Variational Parameters
        mus = self.param(
            "mus", 
            self.parameter_init, 
            (jnp.shape(inputs)[-1] + 1, self.features)) # +1 for bias
        rhos = self.param(
            "rhos",
            self.parameter_init,
            (jnp.shape(inputs)[-1] + 1, self.features) # +1 for bias
        )

        # Sample weights
        weights_and_biases, key = samplevariational_fn(
            mus=mus,
            rhos=rhos,
            key=state["key"],
        )
        # Update state with new key
        state["key"] = key


        weights = weights_and_biases[:-1]
        biases = weights_and_biases[-1]


        y = jnp.dot(inputs, weights) + biases


        log_variational_density = logvariational_fn(
            weights=weights_and_biases,
            mus=mus,
            rhos=rhos,
        )
        log_prior_density = self.logprior(weights)
        
        return (y, log_variational_density, log_prior_density), state
