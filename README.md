# minimally-disruptive-curves

A pure JAX implementation of Minimally Disruptive Curves (MDC). The full user guide is [here](https://dhruva2.github.io/MinimallyDisruptiveCurves.docs/). It is pointed at the Julia implementation, but should still be useful.

MDC finds relationships between parameters that leave a cost function approximately unchanged. It avoids the curse of dimensionality by building out directed curves, rather than an entire space of neutral/sloppy parameters.

Diffrax limitations slightly change the method of evolution from the standard Julia version: there is no discrete 'momentum readjustment' event. I made a continuous approximation of this, and the tests included in the package seem to work. However I haven't tested it as thoroughly as the Julia version, and I predict that the Julia version will build slightly more accurate curves. 

## Features
- **Pure JAX Backend**: Fully differentiable, JIT-compilable, and GPU/TPU ready.
- **Diffrax Integration**: Robust ODE solving under the hood.
- **Transform Chains**: Explore in optimizer space (e.g., log-transforms) and map automatically to physical space.
- **Event Handling**: Terminate curves if cost exceeds momentum or parameters hit bounds.

## Installation

```bash
pip install minimally-disruptive-curves
```
  
## Quickstart

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from minimally_disruptive_curves import MDCProblem, solve_mdc, plot_curve, animate_mdc

# Define a simple cost function
def cost_fn(theta):
    return 0.5 * jnp.sum((theta - jnp.array([1.0, 2.0, 3.0])) ** 2)

# Set up the problem
sys = MDCProblem(
    cost_fn=cost_fn,
    theta0=jnp.array([4.0, 5.0, 6.0]),  # Starting parameters
    dtheta0=jnp.array([1.0, 0.0, 0.0]),  # Initial direction
    momentum=100.0                       # Energy headroom
)

# Solve for the curve
result = solve_mdc(sys, span=(-3.0, 3.0))

# Access the trajectory
print(result.all_t)     # Arc lengths
print(result.all_theta) # Parameter values (N_timesteps, N_params)

# --- Visualization ---

# 1. Static plot of the parameter trajectories
fig, ax = plot_curve(result, raw=True)
plt.show()

# 2. Animate the curve's evolution
# Define a function to draw the live system state on panel 1
def live_sandbox(ax, theta_physical):
    ax.plot(theta_physical, 'go-', markersize=10)
    ax.set_ylim(0, 10)
    ax.set_title("Live Parameters")

anim = animate_mdc(result, live_sandbox, density=50)
anim.save("mdc_curve.gif", fps=10)
 ```

## Using Transform Chains
You can wrap your cost function in a TransformChain:
- Scaling parameters by a constant $c > 1$ will bias the curve to explore those parameters more.
- Fix parameters you aren't interested in
- Log transform parameters if you are interested in relative, not absolute, changes. If you have negative-valued parameters, first make them positive through a scaling transform (scale by -1).


```python
from minimally_disruptive_curves.transforms import ScaleTransform, TransformChain
chain = TransformChain(ScaleTransform(jnp.array([1.0, 1.0, 0.5])))
```
The solver will automatically differentiate through the chain
.
## Citation

There will be a forthcoming software paper you can cite. In the meantime, the algorithm is published 
Raman, Dhruva V., James Anderson, and Antonis Papachristodoulou. "Delineating parameter unidentifiabilities in complex models." Physical Review E 95.3 (2017): 032314.
