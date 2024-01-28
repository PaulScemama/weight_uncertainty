from attrs import define
from functools import partial as bind
from typing import Callable, NamedTuple, Union
from jax import jit, vmap


from optax import GradientTransformation

from typing import Callable, NamedTuple, Tuple, Any, Iterable, Mapping, Union

import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.scipy.stats as stats
import optax
from jax.random import PRNGKey
from optax import GradientTransformation, OptState

from jax.tree_util import tree_leaves, tree_structure, tree_unflatten


"""
This is ported from https://github.com/blackjax-devs/blackjax/blob/main/blackjax/types.py

Following the current best practice (https://jax.readthedocs.io/en/latest/jax.typing.html)
We use:
- `ArrayLike` and `ArrayLikeTree` to annotate function input,
- `Array` and `ArrayTree` to annotate function output.
"""
ArrayTree = Union[jax.Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]
"""-------------------------------------------------------------------------------"""


class MFVIState(NamedTuple):
    mu: ArrayLikeTree
    rho: ArrayLikeTree
    opt_state: OptState


class MFVIInfo(NamedTuple):
    elbo: float
    nll: float
    kl: float


# Useful PyTree Utility: modified from https://github.com/google-research/google-research/blob/master/bnn_hmc/utils/tree_utils.py
# to allow for `n_samples`` to be taken.
def normal_like_tree(a, key, n_samples):
    treedef = tree_structure(a)
    num_vars = len(tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_map(
        lambda p, k: jax.random.normal(k, shape=(n_samples,) + p.shape),
        a,
        tree_unflatten(treedef, all_keys[1:]),
    )
    return noise, all_keys[0]


def iso_gaussian_kl(meanfield_params, wd):
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


# Core functions
def init(
    position: ArrayLikeTree,
    optimizer: GradientTransformation,
) -> MFVIState:
    """Initialize the mean-field VI state"""
    mu = jax.tree_map(jnp.zeros_like, position)
    rho = jax.tree_map(lambda x: 1.0 * jnp.ones_like(x), position)
    opt_state = optimizer.init((mu, rho))
    return MFVIState(mu, rho, opt_state)


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
    key : PRNGKey
        Key for JAX's pseudo-random number generator.
    meanfield_params : ArrayLikeTree
        Values for parameters governing the variational distribution from which to
        sample from.
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
    mu_tree, rho_tree = meanfield_params
    sigma_tree = jax.tree_map(jnp.exp, rho_tree)
    noise_tree, key = normal_like_tree(mu_tree, key, n_samples)
    sample = jax.tree_map(
        lambda mu, sigma, noise: mu + sigma * noise,
        mu_tree,
        sigma_tree,
        noise_tree,
    )
    return sample


def step(
    key: PRNGKey,
    mfvi_state: MFVIState,
    batch: Tuple[jax.Array],
    loglikelihood_fn: Callable,
    kl_fn: Callable,
    optimizer: GradientTransformation,
    n_samples: int,
) -> Tuple[Tuple[float, PRNGKey], ArrayTree]:
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
    logjoint_fn : Callable
        Function mapping data and parameter values to the log probability
        of the joint distribution which represents the probabilistic model.
    n_samples : int
        Number of samples to draw from the variational distribution during the
        computation of the evidence lower bound.

    Returns
    -------
    Tuple[Tuple[float, PRNGKey], ArrayTree]
        The value for the elbo and a fresh pseudo-random number generator, as
        well as the gradient of the elbo with respect to the variational parameters.
    """
    meanfield_params = mfvi_state.mu, mfvi_state.rho

    # negative elbo
    def nelbo(meanfield_params):
        sampled_params = meanfield_sample(key, meanfield_params, n_samples)
        nll = -loglikelihood_fn(sampled_params, batch).mean()  # negative log likelihood
        kl = kl_fn(meanfield_params, sampled_params)  # kl penalty
        elbo = (nll + kl).mean()
        return elbo, (nll, kl)

    # Get elbo and gradients w.r.t variational parameters mu and rho
    nelbo_grad_fn = jax.value_and_grad(nelbo, has_aux=True)

    (nelbo_val, (nll, kl)), nelbo_grad = nelbo_grad_fn(meanfield_params)

    # Update variational parameters and mfvi State and Info
    updates, new_opt_state = optimizer.update(
        nelbo_grad, mfvi_state.opt_state, meanfield_params
    )
    new_mu, new_rho = optax.apply_updates(meanfield_params, updates)
    new_mfvi_state = MFVIState(new_mu, new_rho, new_opt_state)
    return new_mfvi_state, MFVIInfo(nelbo_val, nll, kl)


# INTERFACE -------------------------------------------
class MeanfieldVI(NamedTuple):
    init: Callable
    step: Callable
    sample: Callable


@define
class meanfield_vi:
    """User interface for instantiating a `MeanfieldVI` class, which itself contains
    only `init`, `step`, and `sample` methods. Behind the scenes, the level of
    abstraction that we use for computing the ELBO is:

    ELBO = expected log likelihood + KL between variational posterior and prior

    Construction: There are currently three (3) preferred ways of constructing a
    `MeanfieldVI` instance. We introduce them in order of 'restrictiveness' below...

        (1) `init_w_iso_gauss`: this constructor method, as the name suggests,
            uses an isotropic gaussian log prior. The utility of this is that
            we are then able to compute KL term in the ELBO tractably, thus lowering
            the variance in the gradient estimates as in [2].

        (2) `init_w_logprior_fn`: this constructor method requires the user pass in
            their own logprior function. The generally intractable KL term is then
            approximated as `E[q(w) - p(w)]` as in [1].

        (3) `init_w_kl_fn`: this constructor method requires the user pass in
            their own kl function. This is the most general, and requires the most
            work by the user.

    Every constructor method returns a `MeanfieldVI` instance which one can then
    run inference with.

    References:
        [1] "Weight Uncertainty in Neural Networks" (Blundell et. al) 2015
        [2] "Auto-Encoding Variational Bayes" (Kingma and Welling) 2013
    """

    init = staticmethod(init)
    step = staticmethod(step)
    sample = staticmethod(meanfield_sample)

    # CONSTRUCTORS ---------------------------
    @classmethod
    def _construct(
        cls,
        loglikelihood_fn: Callable,
        kl_fn: Callable,
        optimizer: GradientTransformation,
        n_samples: int,
    ) -> MeanfieldVI:
        # Vmap to apply multiple sets of parameters to the same data.
        loglikelihood_fn = vmap(loglikelihood_fn, in_axes=[0, None])
        init_fn = bind(cls.init, optimizer=optimizer)

        # Convenience for user so they can pass in a `MFVIState` to sample function
        @bind(jit, static_argnames=["n_samples"])
        def sample_fn(key, mfvi_state, n_samples=n_samples):
            meanfield_params = mfvi_state.mu, mfvi_state.rho
            return cls.sample(key, meanfield_params, n_samples)

        step_fn = bind(
            cls.step,
            loglikelihood_fn=loglikelihood_fn,
            kl_fn=kl_fn,
            optimizer=optimizer,
            n_samples=n_samples,
        )
        return MeanfieldVI(init_fn, step_fn, sample_fn)

    @classmethod
    def init_w_iso_gauss(
        cls,
        loglikelihood_fn: Callable,
        optimizer: GradientTransformation,
        n_samples: int,
        weight_decay: float = 1.0,
    ):
        wd = weight_decay
        kl_fn = lambda meanfield_params, _: iso_gaussian_kl(meanfield_params, wd)
        return cls._construct(loglikelihood_fn, kl_fn, optimizer, n_samples)

    @classmethod
    def init_w_logprior_fn(
        cls,
        loglikelihood_fn: Callable,
        logprior_fn: Callable,
        optimizer: GradientTransformation,
        n_samples: int,
    ):
        def kl_fn(meanfield_params, sampled_params):
            variational_logprob = meanfield_logprob(meanfield_params, sampled_params)
            logprior = logprior_fn(sampled_params)
            return variational_logprob - logprior

        return cls._construct(loglikelihood_fn, kl_fn, optimizer, n_samples)

    @classmethod
    def init_w_kl_fn(
        cls,
        loglikelihood_fn: Callable,
        kl_fn: Callable,
        optimizer: GradientTransformation,
        n_samples: int,
    ):
        return cls._construct(loglikelihood_fn, kl_fn, optimizer, n_samples)
