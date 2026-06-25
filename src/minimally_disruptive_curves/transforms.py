# src/minimally_disruptive_curves/transforms.py
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Any, List

class AbstractTransform(eqx.Module):
    """Base class for transforms."""
    def forward(self, x):
        raise NotImplementedError
        
    def inverse(self, y):
        raise NotImplementedError

class ScaleTransform(AbstractTransform):
    w: jnp.ndarray

    def __init__(self, w):
        self.w = jnp.asarray(w)

    def forward(self, x):
        return x * self.w

    def inverse(self, y):
        return y / self.w

class LogAbsTransform(AbstractTransform):
    # Optimizer space (log) -> Physical space (exp)
    def forward(self, x):
        return jnp.exp(x)

    # Physical space -> Optimizer space
    def inverse(self, y):
        return jnp.log(jnp.abs(y))

class FixedParamsTransform(AbstractTransform):
    free_idx: jnp.ndarray
    fixed_idx: jnp.ndarray
    fixed_vals: jnp.ndarray
    full_dim: int

    def __init__(self, free_idx, fixed_vals, full_dim):
        # Convert lists to JAX arrays for safe indexing inside JIT/vjp
        self.free_idx = jnp.asarray(free_idx, dtype=jnp.int32)
        
        fixed_idx = [i for i in range(full_dim) if i not in free_idx]
        self.fixed_idx = jnp.asarray(fixed_idx, dtype=jnp.int32)
        
        self.fixed_vals = jnp.asarray(fixed_vals)
        self.full_dim = full_dim

    def forward(self, x):
        x_full = jnp.zeros(self.full_dim, dtype=x.dtype)
        x_full = x_full.at[self.free_idx].set(x)
        x_full = x_full.at[self.fixed_idx].set(self.fixed_vals)
        return x_full

    def inverse(self, y):
        return y[self.free_idx]


class TransformChain(AbstractTransform):
    ts: List[AbstractTransform]

    def __init__(self, *transforms):
        self.ts = list(transforms)

    def forward(self, x):
        for t in self.ts:
            x = t.forward(x)
        return x

    def inverse(self, y):
        # Apply in reverse order
        for t in reversed(self.ts):
            y = t.inverse(y)
        return y

    def pullback(self, grad_out, y):
        """
        JAX makes this trivial. We use jax.vjp to get the vector-Jacobian 
        product for the entire chain automatically!
        """
        # We need to trace the forward pass to get the pullback function
        def _chain_forward(x):
            return self.forward(x)
        
        # Re-compute the input that generated y (or pass it in if we cached it)
        # For simplicity and correctness in functional JAX, we just re-invert y
        x_input = self.inverse(y)
        
        _, vjp_fn = jax.vjp(_chain_forward, x_input)
        grad_in, = vjp_fn(grad_out)
        return grad_in
