import torch
import torch.nn.functional as F
from basic import (log_variational_per_scalar, 
                         log_variational_per_vector, 
                         sample_variational_scalars, 
                         sample_variational_vectors,
                         log_prior_per_scalar, 
                         log_prior_per_vector,)



class BayesLinear:

    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            prior_pi: float,
            prior_sigma1: float,
            prior_sigma2: float, 
            init_params: torch.Tensor = None
            ):
        
        if init_params:
            self.mus = init_params
            self.rhos = init_params
        else:
            self.mus = torch.empty(size=(in_features, out_features)).normal_()
            self.rhos = torch.empty(size=(in_features, out_features)).normal_()
        
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2

    def __call__(self, x: torch.Tensor, n_samples: int = 1):

        sampled_weights = sample_variational_scalars(
            n_samples=n_samples,
            mus=self.mus.ravel(),
            rhos=self.rhos.ravel(),
        )

        mean_linear_out = torch.stack(
            [F.linear(x, sampled_weights[i].T) for i in range(n_samples)]
        ).mean()

        return mean_linear_out
    
        
        