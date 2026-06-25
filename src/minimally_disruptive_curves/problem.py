# src/minimally_disruptive_curves/problem.py
import equinox as eqx
import jax.numpy as jnp
from typing import Callable
from .transforms import TransformChain

class MDCProblem(eqx.Module):
    """
    Holds the specification for an MDC solve.
    """
    # Mark cost_fn and names as static so JAX bakes them into the compiled graph
    # instead of trying to trace them as arrays.
    cost_fn: Callable = eqx.field(static=True)
    theta0: jnp.ndarray
    dtheta0: jnp.ndarray
    momentum: float
    names: list = eqx.field(static=True)
    chain: TransformChain

    def __init__(self, cost_fn, theta0, dtheta0, momentum, names=None, chain=None):
        self.cost_fn = cost_fn
        self.theta0 = jnp.asarray(theta0)
        self.dtheta0 = jnp.asarray(dtheta0)
        self.momentum = float(momentum)
        
        if names is None:
            self.names = [f"p{i}" for i in range(len(theta0))]
        else:
            self.names = list(names)
            
        self.chain = chain if chain is not None else TransformChain()
