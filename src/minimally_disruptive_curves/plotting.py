# src/minimally_disruptive_curves/plotting.py
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional
from .solver import MDCResult

def animate_mdc(
    result: MDCResult, 
    user_sim_func: Callable, 
    fps: int = 15, 
    density: int = 100,
    raw: bool = True
):
    """
    Create a 3-panel animation of the MDC curve.
    
    Args:
        result: The MDCResult object from solve_mdc.
        user_sim_func: A function(theta_physical) that plots the system behavior 
                       onto the current matplotlib axes (subplot 1).
        fps: Frames per second.
        density: Number of frames to generate.
        raw: If True, map parameters to physical space before plotting.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sys = result.sys
    chain = sys.chain
    
    # 1. Reconstruct the time grid and states
    min_t = float(result.neg_t.min()) if len(result.neg_t) > 0 else 0.0
    max_t = float(result.t.max()) if len(result.t) > 0 else 0.0
    
    # FIX: Safeguard against inf/nan from diffrax early termination
    if not np.isfinite(min_t): min_t = 0.0
    if not np.isfinite(max_t): max_t = 0.0
    if min_t == max_t: max_t = min_t + 1.0 # Prevent linspace crash
    
    full_grid = np.linspace(min_t, max_t, density)
    
    # Evaluate curve at all points
    sampled_states = np.asarray([result(float(t)) for t in full_grid])
    N_params = sampled_states.shape[1]
    
    # Map to physical space if requested
    if raw:
        mapped_states = np.asarray([np.asarray(chain.forward(jnp.array(s))) for s in sampled_states])
        theta0_mapped = np.asarray(chain.forward(sys.theta0))
    else:
        mapped_states = sampled_states
        theta0_mapped = np.asarray(sys.theta0)
        
    # 2. Setup Matplotlib Figure (3 panels)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Minimally Disruptive Curve Evolution")
    
    # Pre-calculate global bounds for panel 3
    # Map to physical space if requested
    if raw:
        mapped_states = np.asarray([np.asarray(chain.forward(jnp.array(s))) for s in sampled_states])
        theta0_mapped = np.asarray(chain.forward(sys.theta0))
    else:
        mapped_states = sampled_states
        theta0_mapped = np.asarray(sys.theta0)
        
    # FIX: Force NaN/Inf to 0.0 to absolutely prevent matplotlib crashes
    mapped_states = np.nan_to_num(mapped_states, nan=0.0, posinf=1e6, neginf=-1e6)
    theta0_mapped = np.nan_to_num(theta0_mapped, nan=0.0, posinf=1e6, neginf=-1e6)
    
    y_min, y_max = mapped_states.min(), mapped_states.max()
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -1.0, 1.0
        
    margin = (y_max - y_min) * 0.1 + 1e-5
    ax3.set_ylim(y_min - margin, y_max + margin)



    margin = (y_max - y_min) * 0.1 + 1e-5
    ax3.set_ylim(y_min - margin, y_max + margin)
    ax3.set_xlim(min_t, max_t)
    ax3.set_title("Continuous Parameter Sweep")
    ax3.set_xlabel("Arc Length (t)")
    
    # Panel 2 setup
    labels = sys.names if raw else [f"opt_{i}" for i in range(N_params)]
    ax2.set_title("Instantaneous Parameter Shift (Δ)")
    ax2.set_ylabel("Deviation from Nominal")
    
    # 3. Animation update function
    def update(frame_idx):
        t_current = full_grid[frame_idx]
        theta_current = mapped_states[frame_idx]
        
        # Clear axes for redrawing
        ax1.clear(); ax2.clear(); ax3.clear()
        
        # --- Panel 1: User Simulation ---
        ax1.set_title("Live System Behavior Profile")
        user_sim_func(ax1, theta_current) # User draws here
        
        # --- Panel 2: Bar Chart ---
        deltas = theta_current - theta0_mapped
        ax2.bar(labels, deltas, color='steelblue', alpha=0.7)
        max_delta = max(np.max(np.abs(mapped_states - theta0_mapped)), 1e-6)
        ax2.set_ylim(-max_delta * 1.2, max_delta * 1.2)
        ax2.axhline(0, color='black', linewidth=0.8)
        
        # --- Panel 3: Trajectory ---
        # Plot full faded lines
        for i in range(N_params):
            ax3.plot(full_grid, mapped_states[:, i], color='gray', alpha=0.2)
            
        # Plot history up to current frame
        for i in range(N_params):
            ax3.plot(full_grid[:frame_idx+1], mapped_states[:frame_idx+1, i], 
                     label=labels[i] if i < len(labels) else f"p{i}")
            
        # Cursor
        ax3.axvline(t_current, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlim(min_t, max_t)
        ax3.set_ylim(y_min - margin, y_max + margin)
        ax3.legend(loc='upper right', fontsize='small')
        
        return []

    anim = FuncAnimation(fig, update, frames=range(density), interval=1000/fps, blit=False)
    # plt.close(fig) # Prevent duplicate display in notebooks
    return anim

def plot_curve(
    result: MDCResult, 
    raw: bool = True,
    max_lines: Optional[int] = None
):
    """
    Create a static plot of the completed MDC curve.
    
    Args:
        result: The MDCResult object from solve_mdc.
        raw: If True, map parameters to physical space before plotting.
        max_lines: If specified, only plot the top N most active parameters.
    """
    import matplotlib.pyplot as plt

    sys = result.sys
    chain = sys.chain
    
    # Extract time and states
    t_grid = np.asarray(result.all_t)
    states = np.asarray(result.all_theta)
    
    # Filter out any inf/nan from early event termination
    valid_mask = np.isfinite(t_grid)
    t_grid = t_grid[valid_mask]
    states = states[valid_mask]
    
    # Map to physical space if requested
    if raw:
        mapped_states = np.asarray([np.asarray(chain.forward(jnp.array(s))) for s in states])
    else:
        mapped_states = states
        
    N_params = mapped_states.shape[1]
    labels = sys.names if raw else [f"opt_{i}" for i in range(N_params)]
    
    # Filter to top movers if max_lines is set
    active_indices = list(range(N_params))
    if max_lines is not None and max_lines < N_params:
        movements = [np.ptp(mapped_states[:, i]) for i in range(N_params)]
        active_indices = sorted(range(N_params), key=lambda i: movements[i], reverse=True)[:max_lines]
        
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for i in active_indices:
        ax.plot(t_grid, mapped_states[:, i], 
                label=labels[i] if i < len(labels) else f"p{i}",
                linewidth=2)
        
    # Add vertical line at t=0 (the starting point)
    ax.axvline(0.0, color='black', linestyle='--', alpha=0.3, label='Start (t=0)')
    
    title_suffix = " (Physical Space)" if raw else " (Optimizer Space)"
    ax.set_title(f"MDC Parameter Trajectories{title_suffix}")
    ax.set_xlabel("Arc Length Path Coordinate (t)")
    ax.set_ylabel("Parameter Value")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig, ax
