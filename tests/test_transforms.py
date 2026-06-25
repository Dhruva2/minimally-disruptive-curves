# tests/test_transforms.py
import jax
import jax.numpy as jnp
import numpy as np
from minimally_disruptive_curves.transforms import (
    ScaleTransform, LogAbsTransform, FixedParamsTransform, TransformChain
)

def test_transform_chain():
    # Optimizer space (2D) -> Fix/Mask (3D) -> Scale (3D) -> LogAbs (3D)
    fpt = FixedParamsTransform(free_idx=[0, 2], fixed_vals=[5.0], full_dim=3)
    st = ScaleTransform([2.0, 1.0, 0.5])
    lat = LogAbsTransform()

    chain = TransformChain(fpt, st, lat)

    x_opt = jnp.array([2.0, 8.0])

    # Manual trace:
    # 1. fpt   -> [2.0, 5.0, 8.0]
    # 2. st    -> [4.0, 5.0, 4.0]
    # 3. lat   -> [exp(4.0), exp(5.0), exp(4.0)]
    expected_y = jnp.array([jnp.exp(4.0), jnp.exp(5.0), jnp.exp(4.0)])
    
    y_final = chain.forward(x_opt)
    np.testing.assert_allclose(y_final, expected_y, rtol=1e-5)

    # Check inverse maps back
    x_reconstructed = chain.inverse(y_final)
    np.testing.assert_allclose(x_reconstructed, x_opt, rtol=1e-5)

    # Test pullback (gradient flow)
    # dy/dx for LogAbs is y. For Scale is w.
    # g_out = [1.0, 1.0, 1.0]
    # g_in_lat = [exp(4.0), exp(5.0), exp(4.0)]
    # g_in_st  = [exp(4.0)*2.0, exp(5.0)*1.0, exp(4.0)*0.5]
    # g_in_fpt = slice free -> [exp(4.0)*2.0, exp(4.0)*0.5]
    expected_g_in = jnp.array([2.0 * jnp.exp(4.0), 0.5 * jnp.exp(4.0)])
    
    g_out = jnp.array([1.0, 1.0, 1.0])
    g_in = chain.pullback(g_out, y_final)
    
    np.testing.assert_allclose(g_in, expected_g_in, rtol=1e-5)
