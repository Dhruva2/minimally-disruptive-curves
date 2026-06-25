# tests/test_transformed_cost.py
import jax
import jax.numpy as jnp
import numpy as np
from minimally_disruptive_curves import MDCProblem, solve_mdc
from minimally_disruptive_curves.transforms import (
    ScaleTransform, LogAbsTransform, FixedParamsTransform, TransformChain
)

def test_transformed_solve():
    """Ensure the solver can trace through a TransformChain correctly."""
    
    # Define a cost function in 3D PHYSICAL space
    center = jnp.array([1.0, 2.0, 3.0])
    def cost_fn(theta_physical):
        return 0.5 * jnp.sum((theta_physical - center) ** 2)

    # Build a chain: Optimizer (2D) -> Mask (3D) -> Scale (3D)
    fpt = FixedParamsTransform(free_idx=[0, 2], fixed_vals=[2.0], full_dim=3)
    st = ScaleTransform([2.0, 1.0, 1.0])
    chain = TransformChain(fpt, st)

    # 1. Figure out the optimizer-space starting point that maps to physical center
    # Physical target: [1.0, 2.0, 3.0]
    # Inverse Scale: [0.5, 2.0, 3.0]
    # Inverse Mask (slice free): [0.5, 3.0]
    theta0_opt = chain.inverse(jnp.array([1.0, 2.0, 3.0]))
    dtheta0_opt = jnp.array([0.1, 0.1])

    sys = MDCProblem(
        cost_fn=cost_fn,
        theta0=theta0_opt,
        dtheta0=dtheta0_opt,
        momentum=100.0,
        chain=chain
    )

    result = solve_mdc(sys, span=(-3.0, 3.0))

    pos_theta = np.asarray(result.theta)
    neg_theta = np.asarray(result.neg_theta)

    # Check that the optimizer space parameters are 2D
    assert pos_theta.shape[1] == 2
    assert neg_theta.shape[1] == 2

    # Start of positive trajectory should match theta0_opt
    np.testing.assert_allclose(pos_theta[0], theta0_opt, atol=1e-1)

    # Check that if we map the start of the trajectory back to physical space,
    # it matches our original physical target
    start_physical = chain.forward(pos_theta[0])
    np.testing.assert_allclose(start_physical, center, atol=1e-1)

if __name__ == "__main__":
    test_transformed_solve()
    print("Transformed solve test passed!")
