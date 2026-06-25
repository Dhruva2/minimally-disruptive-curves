# src/minimally_disruptive_curves/utilities.py
import jax
import jax.numpy as jnp
from typing import Optional, List

def sparse_init_dir(
    hessian: jnp.ndarray, 
    orthogonal_to: Optional[List[jnp.ndarray]] = None, 
    lam: float = 1.0, 
    start: Optional[jnp.ndarray] = None,
    max_iter: int = 2000, 
    tol: float = 1e-6,
    trim_level: float = 1e-5
) -> jnp.ndarray:
    """
    Generates a sparse initial MDC direction.
    """
    n = hessian.shape[0]
    
    # 1. Initialization & Scale Estimation
    E = jnp.linalg.eigh(hessian)
    H_scale = E.eigenvalues[-1]
    effective_lam = lam * H_scale
    t = 1.0 / (2.0 * H_scale)
    
    # Use provided start guess, or default to the first eigenvector
    if start is None:
        x = E.eigenvectors[:, 0]
    else:
        x = start
        
    # Orthogonalize against provided vectors
    if orthogonal_to:
        for v in orthogonal_to:
            x = x - jnp.dot(x, v) * v
        # FIX: Safe normalization to prevent 0/0 -> NaN
        nx = jnp.linalg.norm(x)
        x = jnp.where(nx > 1e-8, x / nx, x)
    else:
        x = x / jnp.linalg.norm(x)

    def body_fn(i, val):
        x = val
        
        # Step A: Gradient Calculation (grad = 2 * H * x)
        grad_smooth = jnp.dot(hessian, x)
        
        # Step B: Gradient Descent + Proximal Operator
        xi = x - t * 2.0 * grad_smooth
        x = jnp.sign(xi) * jnp.maximum(0.0, jnp.abs(xi) - t * effective_lam)
        
        # Step C: Project onto orthogonal subspace
        if orthogonal_to:
            for v in orthogonal_to:
                x = x - jnp.dot(x, v) * v
                
        # Step D: Project back onto unit sphere
        nx = jnp.linalg.norm(x)
        x = jnp.where(nx > 1e-8, x / nx, x)
        
        return x

    # Run the iterative loop
    x_final = jax.lax.fori_loop(0, max_iter, body_fn, x)
    
    # Hard-threshold tiny values
    x_final = jnp.where(jnp.abs(x_final) < trim_level, 0.0, x_final)
    
    # Final re-normalization
    nx = jnp.linalg.norm(x_final)
    x_final = jnp.where(nx > 1e-8, x_final / nx, x_final)
    
    return x_final

def sparse_eigenbasis(
    hessian: jnp.ndarray, 
    num_vectors: int, 
    lam: float = 1.0, 
    max_iter: int = 2000, 
    tol: float = 1e-6
) -> jnp.ndarray:
    """
    Generates a basis of `num_vectors` mutually orthogonal sparse directions.
    """
    n = hessian.shape[0]
    if num_vectors > n:
        raise ValueError("num_vectors cannot exceed the dimension of the Hessian.")
        
    E = jnp.linalg.eigh(hessian)
    basis = []
    
    for i in range(num_vectors):
        # FIX: Pass the i-th eigenvector as the starting guess!
        start_guess = E.eigenvectors[:, i]
        
        v_sparse = sparse_init_dir(
            hessian, 
            orthogonal_to=basis if basis else None, 
            lam=lam, 
            start=start_guess,
            max_iter=max_iter, 
            tol=tol
        )
        basis.append(v_sparse)
        
    return jnp.stack(basis)
