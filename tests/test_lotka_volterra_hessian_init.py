# tests/test_lotka_volterra.py
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import diffrax
from minimally_disruptive_curves import MDCProblem, solve_mdc, make_safety_event

def lotka_volterra_dynamics(t, u, p):
    prey, predator = u
    alpha, beta, gamma, delta = p
    dprey = alpha * prey - beta * prey * predator
    dpred = -delta * predator + gamma * prey * predator
    return jnp.stack([dprey, dpred])

def make_lv_cost(p_nominal, u0=(1.0, 1.0), tspan=(0.0, 10.0), dt=0.1):
    t_eval = jnp.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0]) / dt) + 1)
    u0_jax = jnp.array(u0, dtype=jnp.float64)
    p_nom = jnp.array(p_nominal, dtype=jnp.float64)

    term = diffrax.ODETerm(lotka_volterra_dynamics)
    solver = diffrax.Tsit5()
    
    # Get nominal features (mean prey, max predator)
    sol_nom = diffrax.diffeqsolve(term, solver, t0=tspan[0], t1=tspan[1], dt0=0.05, y0=u0_jax, args=p_nom, saveat=diffrax.SaveAt(ts=t_eval), max_steps=5000)
    nom_features = jnp.array([jnp.mean(sol_nom.ys[:, 0]), jnp.max(sol_nom.ys[:, 1])])

    def cost_fn(p):
        is_invalid = jnp.any(p <= 0.0)
        penalty = 1e6
        
        def solve_ode(_):
            safe_p = jnp.maximum(p, 1e-4)
            solver = diffrax.Tsit5(scan_kind="bounded")
            adjoint = diffrax.DirectAdjoint()
            
            sol = diffrax.diffeqsolve(
                term, solver, t0=tspan[0], t1=tspan[1], dt0=0.05, y0=u0_jax, 
                args=safe_p, saveat=diffrax.SaveAt(ts=t_eval), 
                adjoint=adjoint, max_steps=5000
            )
            
            is_finite = jnp.all(jnp.isfinite(sol.ys))
            current_features = jnp.array([jnp.mean(sol.ys[:, 0]), jnp.max(sol.ys[:, 1])])
            cost = jnp.sum((current_features - nom_features) ** 2)
            return jnp.where(is_finite, cost, penalty)

            
        return jax.lax.cond(is_invalid, lambda _: penalty, solve_ode, None)

    return cost_fn

def test_lotka_volterra_hessian_init():
    """Port of basic_lotka_volterra.jl: Use Hessian eigenvector for init direction."""
    p_nominal = jnp.array([1.5, 1.0, 3.0, 1.0])
    cost_fn = make_lv_cost(p_nominal)

    # Compute Hessian to find insensitive direction
    hess0 = jax.hessian(cost_fn)(p_nominal)
    eigen_decomp = jnp.linalg.eigh(hess0)
    init_dir = eigen_decomp.eigenvectors[:, 0] # smallest eigenvalue

    sys = MDCProblem(
        cost_fn=cost_fn,
        theta0=p_nominal,
        dtheta0=init_dir,
        momentum=1.0,
        names=["alpha", "beta", "gamma", "delta"]
    )

    safety_event = make_safety_event(sys, tol=0.1)
    result = solve_mdc(sys, span=(-1.0, 5.0), events=safety_event)

    # Verify trajectory evolved
    valid_pos_mask = np.isfinite(np.asarray(result.t))
    valid_neg_mask = np.isfinite(np.asarray(result.neg_t))
    
    assert np.sum(valid_pos_mask) > 2
    assert np.sum(valid_neg_mask) > 2

    # Check costs along the curve stay below momentum
    ts = jnp.linspace(0.0, 2.0, 10)
    costs = np.asarray(result.cost_trajectory(ts))
    assert np.all(costs < 1.0)
