import jax
import optax


def build_elbo_train_loop(elbo, opt, model, data):

    def train_loop(
            params,
            model_state,
            opt_state,
            kl_weight
    ):
        xs, ys = data
        (elbo_val, ((nll_val, kl_penalty_val), state)), grad = jax.value_and_grad(elbo, has_aux=True)(
            params,
            model_state,
            model,
            xs,
            ys,
            kl_weight,
        )
        print(f"ELBO: {elbo_val} | MSE: {nll_val} | KL penalty: {kl_penalty_val}")
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, state

    return train_loop
