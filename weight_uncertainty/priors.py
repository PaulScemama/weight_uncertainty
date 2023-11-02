import jax.numpy as jnp
import jax.scipy.stats as stats


class ScaleMixturePrior:
    def __init__(self, pi: float, var1: float, var2: float):
        self.pi = pi
        self.var1 = var1
        self.var2 = var2

    def __call__(self, weights):
        # TODO: just do treemap instead of raveling.
        weights_unraveled = weights.ravel()
        # sqrt(var_) since norm.lopdf expects standard deviation
        gaussian1_log_prob = stats.norm.logpdf(
            weights_unraveled, 0, jnp.sqrt(self.var1)
        )
        gaussian2_log_prob = stats.norm.logpdf(
            weights_unraveled, 0, jnp.sqrt(self.var2)
        )
        return jnp.log(
            self.pi * jnp.exp(gaussian1_log_prob)
            + (1 - self.pi) * jnp.exp(gaussian2_log_prob)
        ).mean()


# def logprior_fn2(params):
#     """Computes the Gaussian prior log-density."""
#     # ToDo izmailovpavel: make temperature treatment the same as in gaussian
#     # likelihood function.
#     n_params = sum([p.size for p in jax.tree_leaves(params)])
#     log_prob = -(0.5 * tree_utils.tree_dot(params, params) * weight_decay +
#                  0.5 * n_params * jnp.log((2 * math.pi) / weight_decay))
#     return log_prob / temperature
