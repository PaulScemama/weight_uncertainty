import torch
import torch.nn.functional as F
from basic import logvariational_fn, samplevariational_fn, logprior_fn
import torch.nn as nn


class BayesLinear(nn.Module):

    """
    Defines a Bayesian Linear Layer.

    Attributes
    ----------
        in_features : int
            number of features in the input
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_pi: float,
        prior_var1: float,
        prior_var2: float,
    ):
        super().__init__()
        # Layer attributes
        self.in_features = in_features
        self.out_features = out_features

        # Parameters governing weights of the layer
        # We add a row for biases
        self.mus = nn.Parameter(
            torch.empty(size=(in_features + 1, out_features)).normal_()
        )
        self.rhos = nn.Parameter(
            torch.empty(size=(in_features + 1, out_features)).normal_()
        )

        # Parameters governing weight's prior distribution
        self.pi = prior_pi
        self.prior_var1 = prior_var1
        self.prior_var2 = prior_var2

    def __call__(self, x, n_samples):
        # For biases
        column_of_ones = torch.ones(x.size(0), 1)
        x_aug = torch.concat([column_of_ones, x], axis=-1)

        sampled_weights = samplevariational_fn(mus=self.mus, rhos=self.rhos)

        # logvariational = logvariational_fn(sampled_weights, self.mus, self.rhos)
        # logprior = logprior_fn(sampled_weights, self.pi, self.var1, self.var2)

        return x_aug @ sampled_weights
