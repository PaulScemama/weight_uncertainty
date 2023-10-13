import jax
import jax.numpy as jnp



# // Negative Log Likelihoods ----------------------------------------
def nll_fn(task: str, yhat, y):
    # Calculate negative log likelihood
    if task == "regression":
        return mse(yhat=yhat, y=y)
    elif task == "classification":
        return cross_entropy(yhat=yhat, y=y)
    else:
        raise ValueError("Tasks are [regression, classification]")
    

def mse(yhat, y):
    return jnp.mean((yhat-y)**2)

def cross_entropy(yhat, y):
    n_classes = yhat.shape[-1]
    y_one_hot = jax.nn.one_hot(y, n_classes)
    nll = jnp.sum(y_one_hot * jax.nn.log_softmax(yhat))