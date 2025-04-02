"""
Contains simple helper functions.

"""

import jax.numpy as jnp
from jax import grad, hessian, vmap, jit
import jax.numpy as jnp
from jax import jit, vmap, tree_map

def del_i(g, argnum=0):
    """
    Partial derivative for a function of signature (d,) ---> ().
    Intended to use when defining PINN loss functions.
    
    """
    def g_splitvar(*args):
        x_ = jnp.array(args)
        return g(x_)

    d_splitvar_di = grad(g_splitvar, argnum)

    def dg_di(x):
        return d_splitvar_di(*x)

    return dg_di


def laplace(func):
    """
    Computes Laplacian via trace of hessian
    
    """
    hesse = hessian(func)
    return lambda x: jnp.sum(jnp.diag(hesse(x)))

def grid_line_search_factory(loss, steps):
    def loss_at_step(step, params, tangent_params):
        updated_params = tree_map(lambda p, dp: p - step * dp, params, tangent_params)
        return loss(updated_params)
        
    v_loss_at_steps = jit(vmap(loss_at_step, (0, None, None)))    

    @jit
    def grid_line_search_update(params, tangent_params):
        losses = v_loss_at_steps(steps, params, tangent_params)
        step_size = steps[jnp.argmin(losses)]
        updated_params = tree_map(
            lambda p, dp: p - step_size * dp, params, tangent_params
        )

        return updated_params, step_size

    return grid_line_search_update