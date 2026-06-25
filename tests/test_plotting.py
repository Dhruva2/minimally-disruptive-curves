# tests/test_plotting.py
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for testing
import jax.numpy as jnp
from minimally_disruptive_curves import MDCProblem, solve_mdc, animate_mdc

def test_animate_mdc():
    """Ensure the animation pipeline runs without crashing."""
    center = jnp.array([1.0, 2.0, 3.0])
    def cost_fn(theta):
        return 0.5 * jnp.sum((theta - center) ** 2)

    sys = MDCProblem(
        cost_fn=cost_fn,
        theta0=jnp.array([4.0, 5.0, 6.0]),
        dtheta0=jnp.array([1.0, 0.0, 0.0]),
        momentum=100.0
    )

    result = solve_mdc(sys, span=(-3.0, 3.0))

    # User defined simulation painter
    # FIX: Must accept (ax, theta) so it knows which axes to draw on
    def mock_sim_func(ax, theta):
        ax.plot(theta, 'go-', markersize=10)
        ax.set_ylim(0, 10) # Fix limits so the panel doesn't jump

    anim = animate_mdc(result, mock_sim_func, density=20)
    
    assert anim is not None
    
    # Save the animation to a file so you can view it!
    anim.save("mdc_test_animation.gif", writer="pillow", fps=10)
    print("\nAnimation saved to mdc_test_animation.gif")
