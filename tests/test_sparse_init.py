# tests/test_sparse_init.py
import jax
import jax.numpy as jnp
import numpy as np
from minimally_disruptive_curves import sparse_init_dir, sparse_eigenbasis

def test_sparse_init_dir():
    # Simple 4x4 diagonal Hessian
    hessian = jnp.diag(jnp.array([10.0, 1.0, 0.1, 5.0]))
    
    # Find the first sparse direction (should target smallest eigenvalue 0.1)
    v = sparse_init_dir(hessian, lam=0.01, max_iter=500)
    
    # Should be normalized
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)
    
    # Should point almost entirely in the direction of index 2 (eigenvalue 0.1)
    np.testing.assert_allclose(np.abs(v[2]), 1.0, atol=1e-2)
    
    # Check orthogonality enforcement
    v1 = sparse_init_dir(hessian, lam=0.01)
    v2 = sparse_init_dir(hessian, orthogonal_to=[v1], lam=0.01)
    
    # Must be orthogonal
    assert abs(np.dot(v1, v2)) < 1e-4

def test_sparse_eigenbasis():
    hessian = jnp.diag(jnp.array([10.0, 1.0, 0.1, 5.0]))
    basis = sparse_eigenbasis(hessian, 3, lam=0.01, max_iter=500)
    
    assert basis.shape == (3, 4)
    
    # Check mutual orthogonality
    for i in range(3):
        for j in range(i+1, 3):
            assert abs(np.dot(basis[i], basis[j])) < 1e-4
