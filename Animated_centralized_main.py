import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

plt.close('all')

# --- Simulation Parameters ---
ap_num = 1000
num_target_points = 5000
T = 30
pause_frames = 10
total_steps = T + pause_frames

# --- Quadrotor Linear Time-Invariant (LTI) Model ---
# State vector: [x, y, vx, vy]
dt = 0.1

A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 0.92, 0],
    [0, 0, 0, 0.92]
])

B = np.array([
    [0.5 * dt**2, 0],
    [0, 0.5 * dt**2],
    [dt, 0],
    [0, dt]
])


# --- Target Distribution Generation ---
def get_text_samples(text, num_points):
    """Convert rendered text into a two-dimensional target point cloud."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    # Use a smaller font size for multiline text.
    font_size = 170 if '\n' in text else 250

    ax.text(
        0.5, 0.5, text,
        fontsize=font_size,
        fontweight='black',
        ha='center',
        va='center',
        family='sans-serif',
        linespacing=0.85
    )

    ax.axis('off')
    fig.canvas.draw()

    # Extract foreground pixels from the rendered text.
    rgba = np.array(fig.canvas.renderer._renderer)
    bw = np.mean(rgba[:, :, :3], axis=2) < 220
    indices = np.argwhere(bw).astype(float)
    h, w = bw.shape

    if len(indices) == 0:
        plt.close(fig)
        return np.zeros((num_points, 2))

    # Map pixel coordinates into the simulation domain.
    if '\n' in text:
        indices[:, 0] = (1 - (indices[:, 0] / h)) * 750 + 125
    else:
        indices[:, 0] = (1 - (indices[:, 0] / h)) * 850 + 75

    indices[:, 1] = (indices[:, 1] / w) * 950 + 25

    plt.close(fig)

    # Sample foreground pixels with replacement.
    return indices[
        np.random.choice(len(indices), num_points, replace=True)
    ][:, [1, 0]]


def get_star_samples(num_points):
    """Generate a star-shaped target distribution with Gaussian noise."""
    t = np.linspace(0, 2 * np.pi, num_points)
    R = 250

    x = R * (np.cos(t) + 0.5 * np.cos(4 * t)) + 500
    y = R * (np.sin(t) - 0.5 * np.sin(4 * t)) + 500

    return np.column_stack([x, y]) + np.random.normal(
        0, 5, (num_points, 2)
    )


# --- Sequential Target Assignment ---
def robust_sequential_assignment(positions, target_samples, ap_num):
    """Assign target locations using sequential capacity allocation."""
    num_targets = len(target_samples)
    target_weights = np.ones(num_targets) / num_targets

    ap_capacity = 1.0 / ap_num
    assigned_ystars = np.zeros((ap_num, 2))
    indices = np.random.permutation(ap_num)

    for i in indices:
        p = positions[i]
        dists = np.sum((target_samples - p)**2, axis=1)
        sorted_idx = np.argsort(dists)

        rem_cap = ap_capacity
        w_sum = np.zeros(2)
        total_w = 0

        for idx in sorted_idx:
            val = target_weights[idx]

            if val <= 1e-15:
                continue

            take = min(rem_cap, val)
            w_sum += target_samples[idx] * take
            target_weights[idx] -= take

            rem_cap -= take
            total_w += take

            if rem_cap <= 1e-12:
                break

        if total_w > 0:
            centroid = w_sum / total_w
            d_to_centroid = np.sum(
                (target_samples - centroid)**2, axis=1
            )
            assigned_ystars[i, :] = target_samples[
                np.argmin(d_to_centroid)
            ]
        else:
            assigned_ystars[i, :] = target_samples[
                np.argmin(dists)
            ]

    return assigned_ystars


# --- Target Distributions !!!!!!!!!!!!!!!!!!!! ---
targets_list = [
    get_text_samples("TTU", num_target_points),
    get_text_samples("IEEE\n TAC", num_target_points),
    get_star_samples(num_target_points)
]


# --- Simulation Initialization ---
states = np.zeros((ap_num, 4))
states[:, :2] = np.random.uniform(50, 950, (ap_num, 2))
states[:, 2:] = np.random.uniform(-15, 15, (ap_num, 2))

all_frames = []


# --- Simulation Execution ---
for target_samples in targets_list:
    # Assign target locations for the current formation.
    ystars = robust_sequential_assignment(
        states[:, :2], target_samples, ap_num
    )

    step_history = []

    for k in range(total_steps):
        if k < T:
            # Proportional position feedback with velocity damping.
            u = (
                15.0 * (ystars - states[:, :2])
                - 7.5 * states[:, 2:]
            )

            # Limit control input magnitudes.
            u = np.clip(u, -500, 500)

            # Propagate each agent through the discrete-time LTI model.
            for i in range(ap_num):
                states[i] = (
                    A @ states[i].reshape(4, 1)
                    + B @ u[i].reshape(2, 1)
                ).flatten()

        else:
            # Stabilize the final formation for display.
            states[:, 2:] *= 0.1
            states[:, :2] = ystars

        # Store position history for animation.
        step_history.append(states[:, :2].copy())
        all_frames.append((
            states[:, :2].copy(),
            target_samples,
            list(step_history)
        ))


# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(10, 11))

ax.set_facecolor('#fdfdfd')
ax.grid(True, linestyle=':', alpha=0.5)
ax.tick_params(axis='both', labelsize=20)

ax.set_xlabel("X Position (m)", fontsize=18, fontweight='bold')
ax.set_ylabel("Y Position (m)", fontsize=18, fontweight='bold')

ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_aspect('equal')

ax.set_title(
    "Quadrotor Swarm Intelligence for Distribution Matching",
    fontsize=20,
    fontweight='bold',
    pad=35,
    color='#2c3e50'
)

# Initialize plot objects once to avoid clearing and redrawing the axes.
target_plot = ax.scatter(
    [], [], s=4, c='gray', alpha=0.1
)

agent_plot = ax.scatter(
    [], [],
    s=25,
    c='royalblue',
    edgecolors='white',
    linewidth=0.5,
    zorder=5
)

# Combine all selected agent trajectories into one collection.
trail_collection = LineCollection(
    [],
    colors='black',
    linewidths=0.6,
    alpha=0.1
)
ax.add_collection(trail_collection)

step_text = ax.text(
    500, 1030, "",
    fontsize=20,
    ha='center',
    color='gray'
)

plt.subplots_adjust(
    top=0.88,
    bottom=0.08,
    left=0.12,
    right=0.95
)


# --- Animation Update ---
def update(f):
    # Retrieve the precomputed state for the current frame.
    pos, tar, hist = all_frames[f]

    # Update existing scatter plots.
    target_plot.set_offsets(tar)
    agent_plot.set_offsets(pos)

    # Update all selected trajectories through a single collection.
    h_np = np.asarray(hist)

    if len(hist) > 1:
        segments = [
            h_np[:, i, :]
            for i in range(0, ap_num, 2)
        ]
        trail_collection.set_segments(segments)
    else:
        trail_collection.set_segments([])

    # Display the exact animation frame index.
    step_text.set_text(f"Current Step: {f}")

    return [
        target_plot,
        agent_plot,
        step_text,
        trail_collection
    ]


# --- Interactive Animation Playback ---
ani = FuncAnimation(
    fig,
    update,
    frames=range(len(all_frames)),
    interval=100,
    repeat=True,
    blit=False,
    cache_frame_data=False
)

# Display the animation without exporting a GIF file.
plt.show()