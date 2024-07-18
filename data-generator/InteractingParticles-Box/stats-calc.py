import numpy as np

def compute_mean_std(file_path):
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    
    all_velocities = []
    all_accelerations = []
    
    for key in data.files:
        ds = data[key]
        positions = ds[0]
        
        # Compute velocities and accelerations for each trajectory
        velocities = positions[1:] - positions[:-1]
        accelerations = positions[2:] - 2 * positions[1:-1] + positions[:-2]
        
        # Flatten the time step and number of particles dimensions
        velocities_flat = velocities.reshape(-1, 2)
        accelerations_flat = accelerations.reshape(-1, 2)
        
        # Append to the global lists
        all_velocities.append(velocities_flat)
        all_accelerations.append(accelerations_flat)
    
    # Concatenate all velocities and accelerations
    all_velocities = np.concatenate(all_velocities, axis=0)
    all_accelerations = np.concatenate(all_accelerations, axis=0)
    
    # Calculate mean and std
    velocity_mean = np.mean(all_velocities, axis=0)
    velocity_std = np.std(all_velocities, axis=0)
    
    acceleration_mean = np.mean(all_accelerations, axis=0)
    acceleration_std = np.std(all_accelerations, axis=0)
    
    return {
        'velocity_mean': velocity_mean,
        'velocity_std': velocity_std,
        'acceleration_mean': acceleration_mean,
        'acceleration_std': acceleration_std
    }


stats = compute_mean_std('train.npz')
print(f"vel_mean: {stats['velocity_mean'].tolist()}")
print(f"vel_std: {stats['velocity_std'].tolist()}")
print(f"acc_mean: {stats['acceleration_mean'].tolist()}")
print(f"acc_std: {stats['acceleration_std'].tolist()}")        
