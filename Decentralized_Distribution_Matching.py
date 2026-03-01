import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Seed for reproducibility
np.random.seed(42)

def wasserstein_distance(pos, target_samples, target_weights, ap_weight):
    """
    Computes the Wasserstein distance, updates target weights, and finds the optimal target point (ystar).
    """
    # Calculate Euclidean distances
    dist_map = np.sqrt(np.sum((target_samples - pos)**2, axis=1))
    # Sort indices by distance
    sorted_indices = np.argsort(dist_map)
    
    W = 0.0
    updated_weights = target_weights.copy()
    ystar = np.zeros(2)
    contrib_weights = np.zeros(len(target_weights))
    
    rem_ap_weight = ap_weight  # Remaining capacity
    for idx in sorted_indices:
        target_val = updated_weights[idx]
        # Determine the amount of mass to capture
        deduct = min(rem_ap_weight, target_val)
        
        W += deduct * dist_map[idx]
        updated_weights[idx] -= deduct
        rem_ap_weight -= deduct
        contrib_weights[idx] += deduct
        
        if rem_ap_weight <= 1e-12:
            break
            
    # Calculate the centroid (ystar)
    if np.sum(contrib_weights) > 0:
        ystar = np.sum(target_samples * contrib_weights[:, np.newaxis], axis=0) / np.sum(contrib_weights)
    else:
        ystar = pos.copy()
        
    return W, updated_weights, ystar

# --- Simulation Parameters ---
ap_num = 30
iterations = 20
T = 50
comm_range = 20  # Communication range for consensus
decay_const = 0.99
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
init_target_weights = np.ones(num_target_points) / num_target_points
agent_target_weights = [init_target_weights.copy() for _ in range(ap_num)]

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
received_weights_log = [None] * iterations

# --- Main Optimization Loop ---
for iter_idx in range(iterations):
    print(f"Iteration: {iter_idx + 1}")
    
    # Initialize log for this iteration
    received_weights_log[iter_idx] = [[None for _ in range(ap_num)] for _ in range(ap_num)]
    
    # 1. Sequential Target Assignment Phase
    for i in range(ap_num):
        pos = sample_positions[i, :]
        _, updated_w, ystar = wasserstein_distance(pos, target_samples, agent_target_weights[i], 1/ap_num)
        agent_target_weights[i] = updated_w
        
        # Min-consensus within comm_range
        for k in range(ap_num):
            for j in range(k+1, ap_num):
                if np.linalg.norm(sample_positions[k,:] - sample_positions[j,:]) < comm_range:
                    # Log received weights and perform consensus
                    received_weights_log[iter_idx][k][j] = agent_target_weights[j].copy()
                    received_weights_log[iter_idx][j][k] = agent_target_weights[k].copy()
                    
                    min_w = np.minimum(agent_target_weights[k], agent_target_weights[j])
                    agent_target_weights[k] = min_w
                    agent_target_weights[j] = min_w
        
        # 2. Control Execution Phase
        U_star = Phi_T.T @ Gamma_T_pinv @ (ystar.reshape(2,1) - np.linalg.matrix_power(A, T) @ pos.reshape(2,1))
        
        curr_x = pos.copy().reshape(2,1)
        for k_step in range(T):
            x_traj[iter_idx*T + k_step, i] = curr_x[0,0]
            y_traj[iter_idx*T + k_step, i] = curr_x[1,0]
            u_val = U_star[k_step, 0]
            curr_x = A @ curr_x + B * u_val
            
        sample_positions[i, :] = curr_x.flatten()

    # 3. Memory-augmented initialization
    # Exchange past information between agents outside comm_range
    for i in range(ap_num):
        new_base_w = init_target_weights.copy()
        if iter_idx > 0:
            memory_list = []
            for j in range(ap_num):
                if i != j:
                    if np.linalg.norm(sample_positions[i,:] - sample_positions[j,:]) > comm_range:
                        past_info = received_weights_log[iter_idx-1][i][j]
                        if past_info is not None:
                            memory_list.append(past_info)
            
            if memory_list:
                memory_min = np.min(np.array(memory_list), axis=0)
                new_base_w += decay_const * (memory_min - init_target_weights)
        
        agent_target_weights[i] = new_base_w

# --- Final Visualization ---
plt.figure(figsize=(10, 8))
plt.scatter(target_samples[:, 0], target_samples[:, 1], s=5, c='g', alpha=0.3, label='Target Distribution')
plt.scatter(init_pos[:, 0], init_pos[:, 1], marker='x', c='b', label='Initial Position')
plt.scatter(sample_positions[:, 0], sample_positions[:, 1], marker='s', c='r', edgecolors='k', label='Final Position')

for i in range(ap_num):
    plt.plot(x_traj[:, i], y_traj[:, i], 'k-', alpha=0.01)

plt.grid(True)
plt.legend()
plt.title("Decentralized Multi-Agent Distribution Matching", fontsize=14)
plt.show()