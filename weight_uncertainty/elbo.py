from typing import Dict, Callable
from weight_uncertainty.likelihoods import mse, cross_entropy
import jax.numpy as jnp



def build_elbo(task: str):

 # Calculate negative log likelihood
    if task == "regression":
        nll_fn = mse
    elif task == "classification":
        nll_fn = cross_entropy
    else:
        raise ValueError("Tasks are [regression, classification]")

    def elbo(
            params: Dict,
            state: Dict,
            model: Callable,  
            x: jnp.array, 
            y: jnp.array, 
            kl_weight: float = 0.01,
        ):
        """
        summary.

        Parameters
        ----------
        params : Dict
            _description_
        model_apply : Callable
            _description_
        key : jax.random.PRNGKey
            _description_
        x : jnp.array
            _description_
        y : jnp.array
            _description_
        task : str, optional
            _description_, by default "regression"
        kl_weight : float, optional
            _description_, by default 0.01

        Returns
        -------
        _type_
            _description_
        """
        # Gather outputs from model
        (logits, log_var_density, log_prior_density), state = model.apply(params, state, x)
        nll = nll_fn(yhat=logits, y=y)
        kl_penalty = kl_weight * (log_var_density - log_prior_density)
        
        # elbo = nll + kl_penalty
        elbo = nll + kl_penalty
        return elbo, ((nll, kl_penalty), state)
    
    return elbo