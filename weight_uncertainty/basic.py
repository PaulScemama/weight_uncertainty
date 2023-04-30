import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
import scipy.stats as stats
import numpy as np
from typeguard import typechecked as typechecker


def rhos_from_sigmas(sigmas):
    return torch.log(torch.exp(sigmas) - 1)


@jaxtyped
@typechecker
def sigmas_from_rhos(rhos: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
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


### PER SCALAR DISTRIBUTION FUNCTIONS ###

@jaxtyped
@typechecker
def log_variational_per_scalar(
    weights: Float[Tensor, "M"],
    mus: Float[Tensor, "M"],
    rhos: Float[Tensor, "M"],
) -> Float[Tensor, "M"]:
    """
    Computes the log density of each individual weight under a normal distribution
    governed by the corresponding (mu, rho).

    Parameters
    ----------
        weights : M-length float tensor
            weights to be evaluated.
        mus : M-length float tensor
            mus each governing an individual normal distribution which
            the corresponding weight will be evaluated under.
        rhos : M-length float tensor
            rhos each governing an individual normal distribution which
            the corresponding wiehgt will be evaluated under.

    Returns
    -------
        M-length float tensor
            the M log probabilities evaluated for each of the Normal(w|mu,rho)
            distributions.

    Notes
    -----
        We define sigmas = log(1 + exp(rhos)) as in section 3.2 of "Weight Uncertainty
        in Neural Networks".
    """
    sigmas = sigmas_from_rhos(rhos)
    return torch.tensor(stats.norm.logpdf(weights, mus, sigmas))



@jaxtyped
@typechecker
def sample_variational_scalars(
    n_samples: int,
    mus: Float[Tensor, "M"],
    rhos: Float[Tensor, "M"],
) -> Float[Tensor, "n_samples M"]:
    """
    Samples from M gaussians governed by the M corresponding mus
    and rhos.

    Parameters
    ----------
        mus : M-length float tensor
            mus each governing an individual normal distribution which
            the corresponding weight will be evaluated under.
        rhos : M-length float tensor
            rhos each governing an individual normal distribution which
            the corresponding weight will be evaluated under.
    Returns
    -------
    [n_samples x M] float tensor
        n_samples of an M-length tensor that represents M samples from M
        independent univariate gaussian distributions.

    Notes
    -----
        We define sigmas = log(1 + exp(rhos)) as in section 3.2 of "Weight Uncertainty
        in Neural Networks"
    """
    M = mus.size(0)
    epsilons = stats.norm.rvs(0, 1, M)
    sigmas = sigmas_from_rhos(rhos)
    samples = []
    for _ in range(n_samples):
        samples.append(mus + sigmas * epsilons)
    return torch.stack(samples)

@jaxtyped
@typechecker
def log_prior_per_scalar(
    weights: Float[Tensor, "M"], pi: float, var1: float, var2: float
) -> Float[Tensor, "M"]:
    """
    Computes the log density of each scalar weight under a scale mixture of
    two zero-mean univariate Gaussians governed by the parameters pi, var1, var2
    (all shared between each weight scalar).

    Parameters
    ----------
    weights : [M] float tensor
        M scalar weights representing the weights of the network to be
        evaluated.
    pi : float
        the proportion of scaling between the mixture of the two Gaussians
    var1 : float
        the variance of the first Gaussian
    var2 : float
        the variance of the second Gaussian

    Returns
    -------
    [M] float tensor 
        the log density of each scalar weight evaluated under the same univariate
        scale mixture prior distribution governed by the parameters pi, var1, and var2.
    """
    gaussian1_log_prob = torch.tensor(stats.norm.logpdf(x=weights, mean=0, cov=var1))
    gaussian2_log_prob = torch.tensor(stats.norm.logpdf(x=weights, mean=0, cov=var2))
    return torch.log(
        pi * torch.exp(gaussian1_log_prob) + (1 - pi) * torch.exp(gaussian2_log_prob)
    )


### PER VECTOR DISTRIBUTION FUNCTIONS ###

@jaxtyped
@typechecker
def log_variational_per_vector(
    weight_vectors: Float[Tensor, "N D"],
    mu_vectors: Float[Tensor, "N D"],
    rho_vectors: Float[Tensor, "N D"],
) -> Float[Tensor, "N"]:
    """
    Computes the log density of each weight vector under a diagonal multivariate gaussian
    distribution governed by the corresponding (mu vector, rho vector).

    Parameters
    ----------
        weight_vectors : [N x D] float tensor
            N D-dimensional vectors representing the weights of the network to be
            evaluated.
        mu_vectors : [N x D] float tensor
            N D-dimensional vectors representing the mean vectors for the
            N independent multivariate gaussian distributions to evaluate
            weights under.
        rho_vectors : [N x D] float tensor
            N D-dimensional vectors representing rho parameters for the N independent
            multivariate gaussian distribution to evaluate weights under.

    Returns
    -------
    [N x 1] float tensor
        the log density for each N weight vector according to the parameters in
        mu_vectors and rho_vectors.

    Notes
    -----
        We define sigmas = log(1 + exp(rhos)) as in section 3.2 of "Weight Uncertainty
        in Neural Networks".
    """
    sigmas = sigmas_from_rhos(rho_vectors)
    covariance_diagonals = sigmas.square()
    # If a 0 is in sigmas -> non-singular since sigmas are diagonals
    if not covariance_diagonals.all():
        raise ValueError(
            f"covariance_diagonals need to all be positive, but they are {covariance_diagonals}"
        )
    # this is from Daniel W https://stackoverflow.com/questions/48686934/numpy-vectorization-of-multivariate-normal
    D = weight_vectors.size(1)
    constant = D * np.log(2 * torch.pi)
    log_determinants = torch.log(torch.prod(covariance_diagonals, axis=1))
    deviations = weight_vectors - mu_vectors
    inverses = 1 / covariance_diagonals
    return -0.5 * (
        constant
        + log_determinants
        + torch.sum(deviations * inverses * deviations, axis=1)
    )


@jaxtyped
@typechecker
def sample_variational_vectors(
    n_samples: int,
    mu_vectors: Float[Tensor, "N D"],
    rho_vectors: Float[Tensor, "N D"],
) -> Float[Tensor, "n_samples N D"]:
    """
    Samples from a N diagonal multivariate gaussians governed by the N mu_vectors
    and N rho_vectors.

    Parameters
    ----------
        n_samples : int
            number of samples to return.
        mu_vectors : [N x D] tensor
            N D-dimensional vectors representing the mean vectors for N independent
            multivariate gaussian distributions.
        rho_vectors : [N x D] tensor
            N D-dimensional vectors representing the rho parameters that's related to
            sigma (see `sigma_from_rhos`).

    Returns
    -------
    n_samples x [N x D] tensor
        n_samples of the [N x D] tensor that represents N samples from N independent
        D-dimensional multivariate gaussian distributions.

    Notes
    -----
        We define sigmas = log(1 + exp(rhos)) as in section 3.2 of "Weight Uncertainty
        in Neural Networks".
    """
    N, D = mu_vectors.size()
    # [N x D] matrix of independent samples each from unit normal
    # TODO: should this be multivariate normal? I.e. N d-dimensional samples from an
    # mvn(0,1)?
    epsilons = stats.norm.rvs(0, 1, (N, D))
    sigmas = sigmas_from_rhos(rho_vectors)
    samples = []
    for _ in range(n_samples):
        samples.append(mu_vectors + sigmas * epsilons)
    return torch.stack(samples).float()


@jaxtyped
@typechecker
def log_prior_per_vector(
    weight_vectors: Float[Tensor, "N D"], pi: float, var1: float, var2: float
) -> Float[Tensor, "N 1"]:
    """
    Computes the log density of each weight vector under a scale mixture of two
    zero-mean multivariate Gaussians governed by the parameters pi, var1, var2
    (all shared between each weight vector).

    Parameters
    ----------
    weight_vectors : [N x D] float tensor
        N D-dimensional vectors representing the weights of the network to be
        evaluated.
    pi : float
        the proportion of scaling between the mixture of the two Gaussians
    var1 : float
        the variance of the first Gaussian
    var2 : float
        the variance of the second Gaussian

    Returns
    -------
    [N x 1] float tensor
        The log density evaluated for each N weight vector under the same D-dimensional
        scale mixture prior distribution governed by the parameters pi, var1, and
        var2.
    """

    D = weight_vectors.size(1)
    gaussian1_log_prob = torch.tensor(
        stats.multivariate_normal.logpdf(
            x=weight_vectors, mean=torch.zeros((D,)), cov=torch.ones((D,)) * var1
        )
    )
    gaussian2_log_prob = torch.tensor(
        stats.multivariate_normal.logpdf(
            x=weight_vectors, mean=torch.zeros((D,)), cov=torch.ones((D,)) * var2
        )
    )
    return torch.log(
        pi * torch.exp(gaussian1_log_prob) + (1 - pi) * torch.exp(gaussian2_log_prob)
    )




