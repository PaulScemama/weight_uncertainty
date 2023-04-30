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
            self.mu_vectors = init_params
            self.rho_vectors = init_params
        else:
            self.mu_vectors = torch.empty(size=(in_features, out_features)).normal_()
            self.rho_vectors = torch.empty(size=(in_features, out_features)).normal_()
        
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2

    def __call__(self, x: torch.Tensor, n_samples: int = 1):

        sampled_weight_vectors = sample_variational_vectors(
            n_samples=n_samples,
            mu_vectors=self.mu_vectors,
            rho_vectors=self.rho_vectors
        )

        mean_linear_out = torch.stack(
            [F.linear(x, sampled_weight_vectors[i].T) for i in range(n_samples)]
        ).mean()

        return mean_linear_out
    
        
        