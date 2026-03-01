import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Seed
np.random.seed(42)

def centralized_wasserstein_assignment(positions, target_samples, target_weights, ap_num):
    """
    Centralized assignment: Controller solves the global OT problem 
    sequentially for all agents using the global state.
    """
    assigned_ystars = np.zeros((ap_num, 2))
    current_target_weights = target_weights.copy()
    ap_weight = 1.0 / ap_num

    # Assignment
    for i in range(ap_num):
        pos = positions[i, :]
        dist_map = np.sqrt(np.sum((target_samples - pos)**2, axis=1))
        sorted_indices = np.argsort(dist_map)
        
        rem_ap_weight = ap_weight
        contrib_weights = np.zeros(len(target_weights))
        
        for idx in sorted_indices:
            target_val = current_target_weights[idx]
            deduct = min(rem_ap_weight, target_val)
            
            current_target_weights[idx] -= deduct
            rem_ap_weight -= deduct
            contrib_weights[idx] += deduct
            
            if rem_ap_weight <= 1e-12:
                break
                
        # Calculate centroid (ystar)
        if np.sum(contrib_weights) > 0:
            assigned_ystars[i, :] = np.sum(target_samples * contrib_weights[:, np.newaxis], axis=0) / np.sum(contrib_weights)
        else:
            assigned_ystars[i, :] = pos.copy()
            
    return assigned_ystars

# --- Simulation Parameters ---
ap_num = 30
iterations = 20
T = 50
num_target_points = 500

# --- Initialization of Agent Positions ---
sample_positions = np.zeros((ap_num, 2))
sample_positions[:, 0] = 500 + 100 * np.random.rand(ap_num)
sample_positions[:, 1] = 2000 + 100 * np.random.rand(ap_num)
init_pos = sample_positions.copy()

# --- Generate Reference Target Distribution (Mixture of 4 Gaussians) ---
a = [0.1, 0.3, 0.4, 0.2]
counts = [int(num_target_points * x) for x in a]
if sum(counts) < num_target_points: 
    counts[-1] += (num_target_points - sum(counts))

t_samples = []
t_samples.append(np.random.multivariate_normal([50, 50], [[200, 0], [0, 200]], counts[0]))
t_samples.append(np.random.multivariate_normal([400, 300], [[300, 0], [0, 700]], counts[1]))
t_samples.append(np.random.multivariate_normal([200, 200], [[500, 0], [0, 100]], counts[2]))
t_samples.append(np.random.multivariate_normal([300, 50], [[100, 20], [20, 200]], counts[3]))
target_samples = 5.0 * np.vstack(t_samples)

# --- Initialize Target Weights ---
# Centralized: Only one global weight map
global_target_weights = np.ones(num_target_points) / num_target_points

# --- System Dynamics and Control Precomputation ---
A = np.array([[0.9, 0.1], [0, 0.9]])
B = np.array([[0], [0.1]])

Phi_T = np.zeros((2, T))
for i in range(1, T + 1):
    Phi_T[:, (i-1):(i)] = np.linalg.matrix_power(A, T-i) @ B

Gamma_T = Phi_T @ Phi_T.T
Gamma_T_pinv = np.linalg.pinv(Gamma_T)

# Logging structures
x_traj = np.zeros((T * iterations, ap_num))
y_traj = np.zeros((T * iterations, ap_num))

# --- Main Optimization Loop ---
for iter_idx in range(iterations):
    print(f"Iteration: {iter_idx + 1}")
    
    # 1. Update targets
    assigned_ystars = centralized_wasserstein_assignment(sample_positions, target_samples, global_target_weights, ap_num)
    
    # 2. Execute control
    for i in range(ap_num):
        pos = sample_positions[i, :]
        ystar = assigned_ystars[i, :]
        x_vec = pos.reshape(2, 1)
        
        U_star = Phi_T.T @ Gamma_T_pinv @ (ystar.reshape(2,1) - np.linalg.matrix_power(A, T) @ x_vec)
        
        curr_x = x_vec.copy()
        for k_step in range(T):
            x_traj[iter_idx*T + k_step, i] = curr_x[0,0]
            y_traj[iter_idx*T + k_step, i] = curr_x[1,0]
            u_val = U_star[k_step, 0]
            curr_x = A @ curr_x + B * u_val
            
        sample_positions[i, :] = curr_x.flatten()

# --- Final Visualization ---
plt.figure(figsize=(10, 8))
plt.scatter(target_samples[:, 0], target_samples[:, 1], s=5, c='g', alpha=0.3, label='Target Distribution')
plt.scatter(init_pos[:, 0], init_pos[:, 1], marker='x', c='b', label='Initial Position')
plt.scatter(sample_positions[:, 0], sample_positions[:, 1], marker='s', c='r', edgecolors='k', label='Final Position')

for i in range(ap_num):
    plt.plot(x_traj[:, i], y_traj[:, i], 'k-', alpha=0.01)

plt.grid(True)
plt.legend()
plt.title("Centralized Multi-Agent Distribution Matching", fontsize=14)
plt.show()