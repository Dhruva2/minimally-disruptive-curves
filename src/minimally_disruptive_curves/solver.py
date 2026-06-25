# src/minimally_disruptive_curves/solver.py
import diffrax
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from typing import Tuple, Optional
from .problem import MDCProblem

class MDCResult(eqx.Module):
    """Result of an MDC solve. Holds the raw diffrax solutions for interpolation."""
    pos_sol: Optional[diffrax.Solution]
    neg_sol: Optional[diffrax.Solution]
    sys: MDCProblem

    @property
    def N(self) -> int:
        return len(self.sys.theta0)

    @property
    def t(self) -> jnp.ndarray:
        return self.pos_sol.ts if self.pos_sol is not None else jnp.array([])
        
    @property
    def theta(self) -> jnp.ndarray:
        if self.pos_sol is not None:
            return self.pos_sol.ys[:, :self.N]
        return jnp.empty((0, self.N))

    @property
    def neg_t(self) -> jnp.ndarray:
        return self.neg_sol.ts if self.neg_sol is not None else jnp.array([])
        
    @property
    def neg_theta(self) -> jnp.ndarray:
        if self.neg_sol is not None:
            return self.neg_sol.ys[:, :self.N]
        return jnp.empty((0, self.N))

    @property
    def all_t(self) -> jnp.ndarray:
        return jnp.concatenate([self.neg_t, self.t])

    @property
    def all_theta(self) -> jnp.ndarray:
        if self.neg_theta.shape[0] > 0:
            return jnp.vstack([self.neg_theta, self.theta])
        return self.theta

    def __call__(self, t: float) -> jnp.ndarray:
        """Evaluate the curve at any arc-length t using continuous interpolation."""
        t = jnp.asarray(t)
        
        if self.pos_sol is not None and self.neg_sol is not None:
            # Evaluate both, but clamp t to their valid bounds [0, t1]
            pos_val = self.pos_sol.evaluate(jnp.maximum(t, 0.0))[:self.N]
            neg_val = self.neg_sol.evaluate(jnp.minimum(t, 0.0))[:self.N]
            # Select the correct one based on the sign of t
            return jnp.where(t >= 0.0, pos_val, neg_val)
        elif self.pos_sol is not None:
            return self.pos_sol.evaluate(jnp.maximum(t, 0.0))[:self.N]
        elif self.neg_sol is not None:
            return self.neg_sol.evaluate(jnp.minimum(t, 0.0))[:self.N]
        
        raise ValueError("Both solutions are empty.")


    def cost_trajectory(self, ts: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Evaluate the cost function along the curve at specific arc-lengths."""
        if ts is None:
            ts = self.all_t
            
        def eval_cost(t):
            theta = self.__call__(t)
            theta_phys = self.sys.chain.forward(theta)
            return self.sys.cost_fn(theta_phys)
            
        return jax.vmap(eval_cost)(ts)


def _mdc_vector_field(t: float, y: jnp.ndarray, args: Tuple) -> jnp.ndarray:
    sys, N, stabilizer_strength = args
    theta = y[:N]
    lam = y[N:]

    def combined_cost(theta_opt):
        theta_phys = sys.chain.forward(theta_opt)
        return sys.cost_fn(theta_phys)

    cost_and_grad = jax.value_and_grad(combined_cost)
    C, grad_cache = cost_and_grad(theta)

    diff_theta = theta - sys.theta0
    dist = jnp.sum(diff_theta ** 2)

    mu2 = (C - sys.momentum) / 2.0
    mu2_smooth = jnp.sign(mu2) * jnp.sqrt(mu2**2 + 1e-20)

    lam_dot_lam = jnp.dot(lam, lam)
    lam_dot_diff = jnp.dot(lam, diff_theta)

    safe_denom = jnp.where(jnp.abs(lam_dot_diff) < 1e-8, 1e-8, lam_dot_diff)
    mu1 = jnp.where(dist > 1e-5, 
                    (lam_dot_lam - 4.0 * mu2**2) / safe_denom, 
                    0.0)
    
    inv_2mu2 = 1.0 / (2.0 * mu2)

    dtheta_unnorm = (-lam + mu1 * diff_theta) * inv_2mu2
    
    dtheta_norm = jnp.linalg.norm(dtheta_unnorm)
    dtheta = jnp.where(dtheta_norm > 1e-8, dtheta_unnorm / dtheta_norm, dtheta_unnorm)

    energy_gap = jnp.maximum(1e-6, sys.momentum - C)
    damping = jnp.dot(lam, dtheta) / energy_gap

    dlam = (mu1 * dtheta - grad_cache) * damping

    if stabilizer_strength > 0.0:
        lambda_target = -2.0 * mu2 * dtheta
        dlam = dlam + stabilizer_strength * (lambda_target - lam)

    return jnp.concatenate([dtheta, dlam])

def make_safety_event(sys: MDCProblem, tol: float = 1e-4):
    """Terminates if cost C approaches or exceeds momentum H."""
    def cond_fn(t, y, args, **kwargs):
        sys, N, _ = args
        theta = y[:N]
        theta_phys = sys.chain.forward(theta)
        C = sys.cost_fn(theta_phys)
        return (sys.momentum - tol) - C

    return diffrax.Event(cond_fn=cond_fn)

def make_bounds_event(lbs: jnp.ndarray, ubs: jnp.ndarray):
    """Terminates if parameters fall outside lower/upper bounds."""
    lbs = jnp.asarray(lbs)
    ubs = jnp.asarray(ubs)

    def cond_fn(t, y, args, **kwargs):
        sys, N, _ = args
        theta = y[:N]
        dist_to_lb = theta - lbs
        dist_to_ub = ubs - theta
        return jnp.min(jnp.concatenate([dist_to_lb, dist_to_ub]))

    return diffrax.Event(cond_fn=cond_fn)






def _solve_single_direction(sys: MDCProblem, u0: jnp.ndarray, t1: float, stabilizer_strength: float, events=None) -> Optional[diffrax.Solution]:
    if t1 == 0.0:
        return None

    term = diffrax.ODETerm(_mdc_vector_field)
    solver = diffrax.Tsit5()
    dt0 = jnp.sign(t1) * 0.01
    saveat = diffrax.SaveAt(steps=True, dense=True)
    
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t1, dt0=dt0, y0=u0, 
        args=(sys,len(sys.theta0), stabilizer_strength),
        saveat=saveat, max_steps=10000, event=events
    )
    return sol


def solve_mdc(
    sys: MDCProblem, 
    span: Tuple[float, float] = (-10.0, 10.0), 
    stabilizer_strength: float = 1.0,
    events=None
) -> MDCResult:
    N = len(sys.theta0)
    
    theta0_physical = sys.chain.forward(sys.theta0)
    C0 = sys.cost_fn(theta0_physical)
    
    dtheta0_norm = jnp.linalg.norm(sys.dtheta0)
    dtheta0_normalized = jnp.where(dtheta0_norm > 1e-8, sys.dtheta0 / dtheta0_norm, sys.dtheta0)
    
    lam0 = (sys.momentum - C0) * dtheta0_normalized
    u0 = jnp.concatenate([sys.theta0, lam0])

    pos_sol = _solve_single_direction(sys, u0, span[1], stabilizer_strength, events)
    neg_sol = _solve_single_direction(sys, u0, span[0], stabilizer_strength, events)

    return MDCResult(pos_sol=pos_sol, neg_sol=neg_sol, sys=sys)
