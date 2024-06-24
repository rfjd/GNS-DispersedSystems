import torch
import numpy as np
import learned_simulator

from absl import flags
from absl import app


# Define the flags and their default values
flag.DEFINE_enum("mode", "train", ["train", "valid", "rollout"], "mode to run the code")

FLAGS = flags.FLAGS
C = 6 # input sequence length

def rollout(simulator: learned_simulator.LearnedSimulator,
            position_sequence: torch.tensor,
            particle_types: torch.tensor,
            num_particles_per_example: torch.tensor,
            num_steps: int):
    """
    Generate a rollout using the first C steps of the position_sequence
    Argumentss:
        simulator (learned_simulator.LearnedSimulator): simulator object
        position_sequence (torch.Tensor): positions of the particles; shape = (num_particles, sequence_length, 2). Here, sequence_length is the total number of time steps of the simulation.
        num_steps (int): sequence length - C
    
    Returns:
        rollout_dict (dict): dictionary containing the initial sequence, predicted and ground truth rollouts, and particle types
        loss: squared error loss between the predicted and ground truth rollouts
    """
    current_position_sequence = position_sequence[:, :C, :]
    predictions = []
    for _ in range(num_steps):
        predicted_position = simulator.predict_position(
            current_position_sequence,
            num_particles_per_example,
            particle_types) # shape = (num_particles, spatial_dimension)
        predictions.append(predicted_position)
        current_position_sequence = torch.cat([current_position_sequence[:, 1:, :], predicted_position[:,None,:]], dim=1)


    predictions = torch.stack(predictions, dim=0) # shape = (num_steps, num_particles, spatial_dimension)
    ground_truth_positions = position_sequence[:, C:, :].permute(1,0,2) # shape = (num_steps, num_particles, spatial_dimension) RF: why permute the ground truvh instead of the predictions?

    
    loss = (ground_truth_positions-predictions)**2

    rollout_dict = {
      'initial_positions': position_sequence[:, :C, :].permute(1, 0, 2).cpu().numpy(),
      'predicted_rollout': predictions.cpu().numpy(),
      'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
      'particle_types': particle_types.cpu().numpy()
    }

    return rollout_dict, loss


