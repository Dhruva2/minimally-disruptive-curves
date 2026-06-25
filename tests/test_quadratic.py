# tests/test_quadratic.py
import jax
import jax.numpy as jnp
import numpy as np
from minimally_disruptive_curves import MDCProblem, solve_mdc

def test_quadratic_cost():
    """Simple quadratic cost: C(theta) = 0.5 * sum((theta - center)^2)."""
    center = jnp.array([1.0, 2.0, 3.0])

    # JAX expects pure functions. We define the cost here.
    def cost_fn(theta):
        return 0.5 * jnp.sum((theta - center) ** 2)

    theta0 = jnp.array([4.0, 5.0, 6.0])
    dtheta0 = jnp.array([1.0, 0.0, 0.0])

    sys = MDCProblem(
        cost_fn=cost_fn,
        theta0=theta0,
        dtheta0=dtheta0,
        momentum=100.0,
        names=["a", "b", "c"]
    )

    result = solve_mdc(sys, span=(-3.0, 3.0))

    # Convert JAX arrays to NumPy for assertions
    pos_t = np.asarray(result.t)
    neg_t = np.asarray(result.neg_t)
    pos_theta = np.asarray(result.theta)
    neg_theta = np.asarray(result.neg_theta)

    # Should have evolved in both directions
    assert len(pos_t) > 1
    assert len(neg_t) > 1

    # Parameter dimension should match
    assert pos_theta.shape[1] == 3
    assert neg_theta.shape[1] == 3

    # Start of positive trajectory should match theta0
    np.testing.assert_allclose(pos_theta[0], theta0, atol=1e-2)

    # Cost should stay below momentum everywhere along the curve
    all_theta = np.asarray(result.all_theta)
    for row in all_theta:
        # Evaluate cost using a non-jit version to avoid tracer errors in the test
         if np.all(np.isfinite(row)):
            c = float(cost_fn(jnp.array(row)))
            assert c < 100.0, f"Cost {c} exceeded momentum 100.0"
