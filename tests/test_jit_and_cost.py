# tests/test_jit_and_cost.py
import jax
import jax.numpy as jnp
import numpy as np
from minimally_disruptive_curves import MDCProblem, solve_mdc

def test_jit_and_cost_trajectory():
    """Ensure solve_mdc can be JIT compiled and cost_trajectory works."""
    center = jnp.array([1.0, 2.0, 3.0])

    def cost_fn(theta):
        return 0.5 * jnp.sum((theta - center) ** 2)

    sys = MDCProblem(
        cost_fn=cost_fn,
        theta0=jnp.array([4.0, 5.0, 6.0]),
        dtheta0=jnp.array([1.0, 0.0, 0.0]),
        momentum=100.0
    )

    # FIX: Mark configuration arguments as static so JAX treats them as compile-time constants
    jit_solve = jax.jit(
        solve_mdc, 
        static_argnames=["span", "stabilizer_strength", "events"]
    )
    
    # The first call compiles, the second call reuses the compiled graph
    result1 = jit_solve(sys, span=(-3.0, 3.0))
    result2 = jit_solve(sys, span=(-3.0, 3.0))

    # Check continuous interpolation at t=1.5
    theta_at_1_5 = result1(1.5)
    assert theta_at_1_5.shape == (3,)

    # Check cost trajectory
    ts = jnp.linspace(-2.0, 2.0, 10)
    costs = result1.cost_trajectory(ts)
    
    assert costs.shape == (10,)
    
    # Cost at t=0 should be the initial cost
    initial_cost = cost_fn(sys.theta0)
    np.testing.assert_allclose(result1.cost_trajectory(jnp.array([0.0]))[0], initial_cost, atol=1e-5)
    
    # All costs should be below momentum
    assert np.all(np.asarray(costs) < 100.0)
