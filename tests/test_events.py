# tests/test_events.py
import jax
import jax.numpy as jnp
import numpy as np
from minimally_disruptive_curves import MDCProblem, solve_mdc, make_bounds_event, make_safety_event
jax.config.update("jax_enable_x64", True)


def test_bounds_termination():
    """Ensure the ODE terminates when parameters hit the specified bounds."""
    
    def cost_fn(theta):
        return 0.5 * jnp.sum(theta ** 2)

    theta0 = jnp.array([0.0, 0.0])
    dtheta0 = jnp.array([1.0, 0.0])
    
    sys = MDCProblem(
        cost_fn=cost_fn,
        theta0=theta0,
        dtheta0=dtheta0,
        momentum=100.0
    )

    lbs = jnp.array([-2.0, -2.0])
    ubs = jnp.array([2.0, 2.0])
    bounds_event = make_bounds_event(lbs, ubs)

    result = solve_mdc(sys, span=(-10.0, 10.0), events=bounds_event)

    # Filter out the 'inf' padding from diffrax
    pos_t = np.asarray(result.t)
    pos_theta = np.asarray(result.theta)
    neg_t = np.asarray(result.neg_t)
    neg_theta = np.asarray(result.neg_theta)

    valid_pos_mask = np.isfinite(pos_t)
    valid_neg_mask = np.isfinite(neg_t)

    valid_pos_t = pos_t[valid_pos_mask]
    valid_pos_theta = pos_theta[valid_pos_mask]
    valid_neg_t = neg_t[valid_neg_mask]
    valid_neg_theta = neg_theta[valid_neg_mask]

    # 1. It should NOT have integrated all the way to 10.0
    assert valid_pos_t.max() < 10.0, "Solver did not terminate early on positive side!"
    assert valid_neg_t.min() > -10.0, "Solver did not terminate early on negative side!"

    # 2. The final positive state should be hovering around the boundary (2.0)
    final_pos_x = valid_pos_theta[-1, 0]
    assert 1.8 <= final_pos_x <= 2.2, f"Positive X ended at {final_pos_x}, expected ~2.0"

    # 3. The final negative state should be hovering around the boundary (-2.0)
    final_neg_x = valid_neg_theta[-1, 0]
    assert -2.2 <= final_neg_x <= -1.8, f"Negative X ended at {final_neg_x}, expected ~-2.0"


def test_safety_termination():
    """Ensure the ODE terminates if cost approaches the momentum limit."""
    
    def cost_fn(theta):
        return 10.0 * jnp.sum(theta ** 2)

    theta0 = jnp.array([0.0, 0.0])
    dtheta0 = jnp.array([1.0, 0.0])
    
    sys = MDCProblem(
        cost_fn=cost_fn,
        theta0=theta0,
        dtheta0=dtheta0,
        momentum=5.0
    )

    safety_event = make_safety_event(sys, tol=0.1)

    result = solve_mdc(sys, span=(-100.0, 100.0), events=safety_event)

    pos_t = np.asarray(result.t)
    pos_theta = np.asarray(result.theta)

    valid_pos_mask = np.isfinite(pos_t)
    valid_pos_t = pos_t[valid_pos_mask]
    valid_pos_theta = pos_theta[valid_pos_mask]

    # Should not have made it anywhere near 100.0
    assert valid_pos_t.max() < 100.0

    # Let's check the cost at the final point
    final_theta = jnp.array(valid_pos_theta[-2])
    final_cost = float(cost_fn(final_theta))
    
    # Cost should be approaching the momentum limit (5.0)
    assert final_cost >= 4.0, f"Cost was {final_cost}, expected to approach 5.0"
