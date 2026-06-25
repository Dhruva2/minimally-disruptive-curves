# tests/test_mass_spring.py
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import diffrax
from minimally_disruptive_curves import MDCProblem, solve_mdc, make_safety_event


def mass_spring_dynamics_diffrax(t, u, theta):
    """du/dt for mass-spring-damper. u = [position, velocity], theta = [m, c, k]"""
    x, v = u
    m, c, k = theta
    m = jnp.maximum(m, 1e-3)
    c = jnp.maximum(c, 1e-3)
    k = jnp.maximum(k, 1e-3)
    return jnp.stack([v, -(c / m) * v - (k / m) * x])

def make_mass_spring_cost(theta_nominal, u0=(1.0, 0.0), tspan=(0.0, 5.0), dt=0.5):
    t_eval = jnp.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0]) / dt) + 1)
    u0_jax = jnp.array(u0, dtype=jnp.float64)
    theta_nom = jnp.array(theta_nominal, dtype=jnp.float64)

    term = diffrax.ODETerm(mass_spring_dynamics_diffrax)
    solver = diffrax.Tsit5()

    # Solve nominal trajectory once
    sol_nom = diffrax.diffeqsolve(
        term, solver, t0=tspan[0], t1=tspan[1], dt0=0.1, 
        y0=u0_jax, args=theta_nom, saveat=diffrax.SaveAt(ts=t_eval), max_steps=1000
    )
    target = sol_nom.ys

    def cost_fn(theta):
        is_invalid = jnp.any(theta <= 1.0e-3)
        penalty = 100.0 + jnp.sum(jnp.minimum(0.0, theta) ** 2)
        
        def solve_and_eval(_):
            safe_theta = jnp.maximum(theta, 1.0e-3)
            adjoint = diffrax.RecursiveCheckpointAdjoint()
            sol = diffrax.diffeqsolve(
                term, solver, t0=tspan[0], t1=tspan[1], dt0=0.1, y0=u0_jax, 
                args=safe_theta, saveat=diffrax.SaveAt(ts=t_eval), 
                adjoint=adjoint, max_steps=1000
            )
            is_finite = jnp.all(jnp.isfinite(sol.ys))
            cost = jnp.mean((sol.ys - target) ** 2)
            return jnp.where(is_finite, cost, penalty)
        
        # FIX: Use lax.cond to short-circuit and avoid computing NaN gradients
        return jax.lax.cond(is_invalid, lambda _: penalty, solve_and_eval, None)


    return cost_fn

def test_mass_spring_invariants():
    """The MDC should trace along the ratio-preserving direction (c/m, k/m constant)."""
    theta_nominal = jnp.array([1.0, 0.5, 5.0])  # m, c, k

    cost_fn = make_mass_spring_cost(theta_nominal)

    # Initial direction matches the structural unidentifiability
    dtheta0 = jnp.array([1.0, 0.5, 5.0])

    sys = MDCProblem(
        cost_fn=cost_fn,
        theta0=theta_nominal,
        dtheta0=dtheta0,
        momentum=1.0,
        names=["mass", "damping", "stiffness"]
    )

    # FIX: Add safety event to stop before C approaches H=1.0 and causes a singularity
    safety_event = make_safety_event(sys, tol=0.1)

    result = solve_mdc(sys, span=(-3.0, 3.0), stabilizer_strength=1.0, events=safety_event)

    pos_t = np.asarray(result.t)
    neg_t = np.asarray(result.neg_t)
    
    # Filter out inf padding from early termination
    valid_neg_mask = np.isfinite(neg_t)
    neg_theta = np.asarray(result.neg_theta)[valid_neg_mask]

    assert len(pos_t) > 1
    assert len(neg_theta) > 1

    final_theta = jnp.array(neg_theta[-1])

    initial_cm = theta_nominal[1] / theta_nominal[0]
    initial_km = theta_nominal[2] / theta_nominal[0]
    final_cm = final_theta[1] / final_theta[0]
    final_km = final_theta[2] / final_theta[0]

    # Invariant preservation: c/m and k/m should be constant along the curve
    np.testing.assert_allclose(final_cm, initial_cm, rtol=1e-2)
    np.testing.assert_allclose(final_km, initial_km, rtol=1e-2)

    # Cost at the end should be near zero
    final_cost = float(cost_fn(final_theta))
    assert final_cost < 1e-3, f"Cost too high at curve end: {final_cost}"
