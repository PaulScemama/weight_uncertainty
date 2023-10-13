from flax import linen as nn 
from flax.linen.module import compact
from typing import Callable, Dict, List
import jax.numpy as jnp
from weight_uncertainty.variational import MeanFieldLinearLayer

class MeanFieldNN(nn.Module):

    hidden_features: List[int]
    logprior: Callable
    parameter_init: Callable = nn.initializers.lecun_normal()

    @compact
    def __call__(self, state: Dict, inputs: jnp.array):
        x = inputs
        total_log_variational_density = 0.0
        total_log_prior_density = 0.0
    
        for i, feat in enumerate(self.hidden_features):
            (x, log_variational_density, log_prior_density), state = MeanFieldLinearLayer(feat, self.logprior)(state, x)
            
            total_log_variational_density += log_variational_density
            total_log_prior_density += log_prior_density
            
            if i != len(self.hidden_features) - 1:
                x = nn.relu(x)
        
        return (x, total_log_variational_density, total_log_prior_density), state
