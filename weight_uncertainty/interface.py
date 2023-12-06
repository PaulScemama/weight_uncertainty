from functools import partial
from typing import Callable, NamedTuple

from jax import jit, vmap
from jax.random import PRNGKey
from optax import GradientTransformation

from weight_uncertainty.core import *
from weight_uncertainty.kl import *
from weight_uncertainty.types import ArrayLikeTree


class MeanfieldVI(NamedTuple):
    init: Callable
    step: Callable
    sample: Callable


# Interface
class meanfield_vi:
    init = staticmethod(init)
    step = staticmethod(step)
    sample = staticmethod(meanfield_sample)

    def __new__(
        cls,
        loglikelihood_fn: Callable,
        optimizer: GradientTransformation,
        n_samples: int,
        logprior_fn: Callable = None,
        logprior_name: str = None,
        weight_decay: float = None,
    ):
        kl_fn = _create_kl_fn(logprior_fn, logprior_name, weight_decay)
        loglikelihood_fn = vmap(loglikelihood_fn, in_axes=[0, None])

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, optimizer)

        @jit
        def step_fn(
            key: PRNGKey,
            mfvi_state: MFVIState,
            batch: ArrayLikeTree,
        ):
            return cls.step(
                key, mfvi_state, batch, loglikelihood_fn, kl_fn, optimizer, n_samples
            )

        @partial(jit, static_argnames=["n_samples"])
        def sample_fn(
            key: PRNGKey,
            mfvi_state: MFVIState,
            n_samples: int,
        ):
            meanfield_params = mfvi_state.mu, mfvi_state.rho
            return cls.sample(key, meanfield_params, n_samples)

        return MeanfieldVI(init_fn, step_fn, sample_fn)


def _create_kl_fn(logprior_fn, logprior_name, weight_decay):
    if (logprior_fn is None) == (logprior_name is None):
        raise ValueError(
            "Either `logprior_fn` or `logprior_name` must be specified, but not both."
        )

    if logprior_name:
        if logprior_name == "unit_gaussian":
            if weight_decay is not None:
                print(
                    f"Warning: ignoring `weight_decay` argument because 'unit_gaussian' prior doesn't take it as input."
                )
            return lambda meanfield_params, sampled_params: unit_gaussian_kl(
                meanfield_params
            )

        elif logprior_name == "isotropic_gaussian":
            if weight_decay is None:
                raise ValueError(
                    "`weight_decay` must be specified if using isotropic gaussian prior."
                )
            return lambda meanfield_params, sampled_params: partial(
                isotropic_gaussian_kl, wd=weight_decay
            )(meanfield_params)

        else:
            raise ValueError(
                f"`logprior_name` must be either 'unit_gaussian' or 'isotropic_gaussian' but is {logprior_name}."
            )

    # if logprior_fn
    else:

        def kl_fn(meanfield_params, sampled_params):
            return meanfield_logprob(meanfield_params, sampled_params) - logprior_fn(
                sampled_params
            )

        return kl_fn
