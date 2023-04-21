import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
import scipy
import scipy.stats as stats
import numpy as np
from typeguard import typechecked as typechecker



def rhos_from_sigmas(sigmas):
    return torch.log(torch.exp(sigmas) - 1)

@jaxtyped
@typechecker
def sigmas_from_rhos(rhos: Float[Tensor, "N D"]) -> Float[Tensor, "N D"]:
    """
    Parameterization of the standard deviation sigma. See section 3.2 of 
    "Weight Uncertainty in Neural Networks"

    Parameters
    ----------
    rhos : N x D float tensor
        N D-dimensional vectors representing rho parameters for the N independent
        multivariate gaussian distributions.

    Returns
    -------
    sigmas : N x D float tensor
        N D-dimensional vectors representing sigma parameters for the N independent
        multivariate gaussian distributions.
    """
    return torch.log(1 + torch.exp(rhos))

@jaxtyped
@typechecker
def logvariational_fn(
    weights: Float[Tensor, "N D"],
    mus: Float[Tensor, "N D"],
    rhos: Float[Tensor, "N D"],
) -> Float[Tensor, "N"]:
    """
    Computes the log density of the weights under a diagonal multivariate gaussian 
    distribution governed by (mus, rhos) where sigmas = log(1 + exp(rhos)). See 
    section 3.2 of "Weight Uncertainty in Neural Networks"

    Parameters
    ----------
        weights : N x D float tensor
            N D-dimensional vectors representing the weights of the network to be
            evaluated.
        mus : N x D float tensor
            N D-dimensional vectors representing the mean vectors for the
            N independent multivariate gaussian distributions to evaluate
            weights under.
        rhos : N x D float tensor
            N D-dimensional vectors representing rho parameters for the N independent
            multivariate gaussian distribution to evaluate weights under.

    Returns
    -------
    N x 1 float tensor
        the log density for each N weight vectors according to the parameters in
        mus and rhos.
    """
    sigmas = sigmas_from_rhos(rhos)
    covariance_diagonals = sigmas.square()
    # If a 0 is in sigmas -> non-singular since sigmas are diagonals
    if not covariance_diagonals.all():
        raise ValueError(f"covariance_diagonals need to all be positive, but they are {covariance_diagonals}")
    # this is from Daniel W https://stackoverflow.com/questions/48686934/numpy-vectorization-of-multivariate-normal
    D = weights.size(1)
    constant = D * np.log(2 * torch.pi)
    log_determinants = torch.log(torch.prod(covariance_diagonals, axis=1)) 
    deviations = weights - mus
    inverses = 1 / covariance_diagonals
    return -0.5 * (constant + log_determinants +
        torch.sum(deviations * inverses * deviations, axis=1))


@jaxtyped
@typechecker
def samplevariational_fn(
    n_samples: int,
    mus: Float[Tensor, "N D"],
    rhos: Float[Tensor, "N D"],
) -> Float[Tensor, "N D"]:
    """
    Samples from a diagonal multivariate gaussian governed by the mus
    and sigmas.

    Parameters
    ----------
        n_samples : int
            number of samples to return.
        mus : N x D tensor
            N D-dimensional vectors representing the mean vectors for N independent
            multivariate gaussian distributions.
        rhos : N x D tensor
            N D-dimensional vectors representing the rho parameters that's related to
            sigma (see `sigma_from_rhos`).

    Returns
    -------
    n_samples x N x D tensor
        n_samples of the N x D tensor that represents N samples from N independent
        D-dimensional multivariate gaussian distributions.
    """
    N, D = mus.size()
    # N x D matrix of independent samples each from unit normal
    # TODO: should this be multivariate normal? I.e. N d-dimensional samples from an 
    # mvn(0,1)? 
    epsilons = stats.norm.rvs(0, 1, (N, D))
    sigmas = sigmas_from_rhos(rhos)
    samples = []
    for _ in range(n_samples):
        samples.append(mus + sigmas * epsilons)
    return torch.stack(samples)

@jaxtyped
@typechecker
def logprior_fn(weights: Float[Tensor, "N D"], pi: float, sigma1: float, sigma2: float):
    print(weights.shape)

    gaussian1_log_prob = stats.multivariate_normal.logpdf(
        x=weights, mean=0, cov=sigma1**2
    )
    gaussian2_log_prob = stats.multivariate_normal.logpdf(
        x=weights, mean=0, cov=sigma2**2
    )

    return torch.log(
        pi * torch.exp(gaussian1_log_prob) + (1 - pi) * torch.exp(gaussian2_log_prob)
    )
