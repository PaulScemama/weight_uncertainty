from collections import abc
from functools import partial, singledispatch
from typing import Callable, NamedTuple, Union
from jax import jit, vmap
from jax.random import PRNGKey

from weight_uncertainty.core import *
from weight_uncertainty.kl import *
from weight_uncertainty.types import ArrayLikeTree
import warnings
from optax import GradientTransformation


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
        logprior: Union[Callable, str],
        optimizer: GradientTransformation,
        n_samples: int,
        weight_decay: float = None,
    ):
        kl_fn = _create_kl_fn(logprior, weight_decay)
        loglikelihood_fn = vmap(loglikelihood_fn, in_axes=[0, None])

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, optimizer)

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


# ------------------------ Helpers --------------------------------------------------


def _approx_kl_fn(logprior_fn):
    def kl(meanfield_params, sampled_params):
        return meanfield_logprob(meanfield_params, sampled_params) - logprior_fn(
            sampled_params
        )

    return kl


@singledispatch
def _create_kl_fn(
    logprior: Union[Callable, str],
    weight_decay=None,
):
    raise ValueError(f"logprior must be Callable or str, got f{type(logprior)}")


@_create_kl_fn.register(str)
def _(logprior: str, weight_decay=None):
    if logprior == "unit_gaussian":
        return lambda meanfield_params, _: unit_gaussian_kl(meanfield_params)

    elif logprior == "isotropic_gaussian":
        kl_fn = partial(isotropic_gaussian_kl, wd=weight_decay)
        return lambda meanfield_params, _: kl_fn(meanfield_params)

    else:
        raise ValueError(
            "Currently only 'unit_gaussian' and 'isotropic_gaussian' are registered."
        )


@_create_kl_fn.register
def _(logprior: abc.Callable, weight_decay=None):
    return _approx_kl_fn(logprior)
