import torch

def get_random_walk_noise_for_position_sequence(position_sequence: torch.tensor,
                                                noise_std_last_step: float) -> torch.tensor:
    """
    Returns a noise sequence for the current position sequence (length: C). A noise sequence is first generated for the corresponding velocity sequence (length: C-1), while ensuring that the cumulative noise of the random walk after C-1 steps is equal to the noise_std_last_step. The output position_sequence_noise is then computed by integrating the cumulative velocity_sequence_noise.

    Args: 
        position_sequence: current sequence of particle positions; shape = (num_particles, C, spatial_dimension)
        noise_std_last_step: standard deviation of the radnom walk noise in the last step

    Returns:
        random_walk_position_sequence_noise: the noise in the position sequence.
    """
    velocity_sequence = torch.diff(position_sequence, dim=1) # assuming dt=1; note that only the shape (and not the values) of the velocity_sequence is used in this function
    num_velocities = velocity_sequence.shape[1] # C-1; number of random walk steps
    std_each_step = noise_std_last_step/num_velocities**0.5 # Variance of sum of independent random variables is the sum of their variances: \mathrm{Var}(v_1, v_2, ..., v_C) = \sum_{i=1}^{C} \mathrm{Var}(v_i) \Rightarrow \sigma_{\text{each step}} = \frac{\sigma_{\text{last step}}}{\sqrt{C-1}}
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape))*std_each_step

    # Apply the random walk
    random_walk_velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    # Integrate (dt=1) the noise in the velocity to the positions
    random_walk_position_sequence_noise = torch.cat([torch.zeros_like(random_walk_velocity_sequence_noise[:, 0:1]), torch.cumsum(random_walk_velocity_sequence_noise, dim=1)], dim=1)

    return random_walk_position_sequence_noise



