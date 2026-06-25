# tests/test_mass_spring_transforms.py
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import diffrax
from minimally_disruptive_curves import MDCProblem, solve_mdc, make_safety_event
from minimally_disruptive_curves.transforms import LogAbsTransform, FixedParamsTransform, TransformChain

def mass_spring_dynamics(t, u, theta):
    x, v = u
    m, c, k = theta
    m = jnp.maximum(m, 1e-3)
    c = jnp.maximum(c, 1e-3)
    k = jnp.maximum(k, 1e-3)
    return jnp.stack([v, -(c / m) * v - (k / m) * x])

def make_mse_cost(theta_nominal, u0=(1.0, 0.0), tspan=(0.0, 10.0), dt=0.2):
    t_eval = jnp.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0]) / dt) + 1)
    u0_jax = jnp.array(u0, dtype=jnp.float64)
    theta_nom = jnp.array(theta_nominal, dtype=jnp.float64)

    term = diffrax.ODETerm(mass_spring_dynamics)
    solver = diffrax.Tsit5()
    sol_nom = diffrax.diffeqsolve(term, solver, t0=tspan[0], t1=tspan[1], dt0=0.1, y0=u0_jax, args=theta_nom, saveat=diffrax.SaveAt(ts=t_eval), max_steps=1000)
    target = sol_nom.ys[:, 0] # just position

    def cost_fn(theta):
        is_invalid = jnp.any(theta <= 1.0e-3)
        penalty = 100.0 + jnp.sum(jnp.minimum(0.0, theta) ** 2)
        
        def solve_ode(_):
            safe_theta = jnp.maximum(theta, 1.0e-3)
            adjoint = diffrax.RecursiveCheckpointAdjoint()
            sol = diffrax.diffeqsolve(term, solver, t0=tspan[0], t1=tspan[1], dt0=0.1, y0=u0_jax, args=safe_theta, saveat=diffrax.SaveAt(ts=t_eval), adjoint=adjoint, max_steps=1000)
            return jnp.mean((sol.ys[:, 0] - target) ** 2)
            
        return jax.lax.cond(is_invalid, lambda _: penalty, solve_ode, None)

    return cost_fn

def test_mass_spring_transforms():
    """Port of mass_spring_transforms.jl: Fix mass, explore log(c) and log(k)."""
    theta_nominal = jnp.array([1.0, 0.5, 5.0]) # m, c, k
    core_cost = make_mse_cost(theta_nominal)

    # Chain: [log(c), log(k)] -> [c, k] -> [1.0, c, k]
    fix_transform = FixedParamsTransform(free_idx=[1, 2], fixed_vals=[1.0], full_dim=3)
    chain = TransformChain(LogAbsTransform(), fix_transform)

    # theta0 in optimizer space
    theta0_opt = chain.inverse(theta_nominal)
    dtheta0_opt = jnp.array([1.0, 1.0])

    sys = MDCProblem(
        cost_fn=core_cost,
        theta0=theta0_opt,
        dtheta0=dtheta0_opt,
        momentum=1.0,
        names=["mass", "damping", "stiffness"],
        chain=chain
    )

    safety_event = make_safety_event(sys, tol=0.1)
    result = solve_mdc(sys, span=(-5.0, 5.0), events=safety_event)

    # Extract final state and map back to physical space
    valid_neg_mask = np.isfinite(np.asarray(result.neg_t))
    final_opt = jnp.array(np.asarray(result.neg_theta)[valid_neg_mask][-1])
    final_physical = chain.forward(final_opt)

    # Mass should stay completely fixed at 1.0
    assert final_physical[0] == 1.0, "Mass did not stay fixed!"
    
    # Cost at the end should be very low
    final_cost = float(core_cost(final_physical))
    assert final_cost < 1.0, f"Cost too high: {final_cost}"
