from gns import seed_util
seed_util.apply_seed()
from gns import network_architecture

import torch
import numpy as np
import random

import torch.nn as nn

import torch_geometric.nn as gnn
from typing import Dict, Tuple

class LearnedSimulator(nn.Module):
    """
    The constructor gets the following arguments:
    
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 num_message_passing_steps: int,
                 connectivity_radius: float,
                 # normalization_stats: Dict['str', Dict['str', torch.tensor]],
                 boundaries: np.ndarray, # shape = (spatial_dimension, 2): lo and hi corners of the simulation box
                 rotation: bool = False,
                 spatial_dimension: int = 2,
                 num_encoded_node_features: int = 128,
                 num_encoded_edge_features: int = 128,
                 num_mlp_layers: int = 2,
                 mlp_layer_size: int = 128,
                 device="cpu",
                 rMax: float = 4.0):
        super().__init__()
        self.connectivity_radius = connectivity_radius
        self.boundaries = boundaries
        self.rMax = rMax
        
        self.output_node_size = spatial_dimension + 1 if rotation else spatial_dimension
        self.encoder_processor_decoder = network_architecture.EncoderProcessorDecoder(
            num_node_features,
            num_encoded_node_features,
            num_edge_features,
            num_encoded_edge_features,
            num_mlp_layers,
            mlp_layer_size,
            num_message_passing_steps,
            self.output_node_size)
        self.device = device

    def get_particle_radii(self, particle_properties: torch.tensor):
        if len(list(particle_properties.shape)) == 1:
            particle_radii = particle_properties.view(-1, 1) # shape = (num_particles,)
        else:
            particle_radii = particle_properties[:,0].view(-1, 1) # shape = (num_particles,)

        return particle_radii
    
    def forward(self):
        pass

    def compute_graph_connectivity(self,
                                   particle_locations: torch.tensor,
                                   num_particles_per_example: torch.tensor,
                                   add_self_edges: bool = True):
        """
        This method generates edges between particles that are within the connectivity_radius of each other.

        Arguments:
            particle_locations: shape = (num_particles, spatial_dimension)
            num_particles_per_example: number of particles per example in the batch; we usually have 2 examples per batch. Hence num_particles_per_example = tensor([num_particles_example1, num_particles_example2]).
            add_self_edges: whether to add self edges (loops) or not
        
        Returns:
            receivers and senders of edge_index [edge_index.shape = (2, num_edges)]; the num_edges corresponds to the total number of edges in the batch (i.e., between num_particles). The batch_ids below is used to distinguish between particles from different examples in the batch.
        """
        # bacth_ids: tensor containing batch indices for each particle. It's constructed using torch.cat to repeat batch indices based on nparticles_per_example. For instance, if num_particles_per_example = tensor([3,4]), then batch_ids = tensor([0,0,0,1,1,1,1]).
        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(num_particles_per_example)]).to(self.device)

        edge_index = gnn.radius_graph(particle_locations, r=self.connectivity_radius, batch=batch_ids, loop=add_self_edges, max_num_neighbors=128)
        receivers = edge_index[0, :]
        senders = edge_index[1, :]

        return receivers, senders

    def encoder_preprocessor(self,
                             position_sequence: torch.tensor,
                             num_particles_per_example: torch.tensor,
                             particle_properties: torch.tensor):
        """
        This method encodes the particle positions and properties using a sequence of C=6 most recent positions.

        Arguments:
            position_sequence: tensor of shape (num_particles, C, spatial_dimension)
            num_particles_per_example: tensor of shape (batch_size,) containing the number of particles in each example
            particle_properties: tensor of shape (num_particles, N) containing N properties of each particle

        Returns:
            node_features: tensor of shape (num_particles, num_node_features)
            edge_features: tensor of shape (num_edges, num_edge_features)
            receivers, senders: tensors of shape (num_edges,) containing the receiver and sender indices of each edge
        """
      
        particle_radii = self.get_particle_radii(particle_properties) # shape = (num_particles, 1)
        num_particles = position_sequence.shape[0]
        current_position = position_sequence[:, -1, :] # last position: shape = (num_particles, spatial_dimension)
        last_velocities = torch.diff(position_sequence, dim=1) # last C-1 velocities: shape = (num_particles, C-1, spatial_dimension)

        ### Encoded node features
        # flattened most recent velocities
        node_features = []
        
        normalized_velocities = last_velocities/particle_radii.unsqueeze(-1) # shape = (num_particles, C-1, spatial_dimension)
        flat_normalized_velocities = normalized_velocities.view(num_particles, -1) # shape = (num_particles, (C-1)*spatial_dimension)

        node_features.append(flat_normalized_velocities)

        # norlamized distance to boundaries
        boundaries = torch.tensor(self.boundaries, requires_grad=False).to(self.device) # converting boundaries from np.ndarray to torch.tensor; shape = (spatial_dimension, 2); boundaries[:,0] and boundaries[:,1] give the coordinates of the low and high corners of the simulation box with shape (spatial_dimension,), respectively.
        distannce_to_lower_boundary = current_position - boundaries[:, 0] # shape = (num_particles, spatial_dimension);
        distannce_to_upper_boundary = boundaries[:, 1] - current_position # shape = (num_particles, spatial_dimension);
        distance_to_boundaries = torch.cat([distannce_to_lower_boundary, distannce_to_upper_boundary], dim=-1)
        
        normalized_distance_to_boundaries = distance_to_boundaries/particle_radii # shape = (num_particles, 2*spatial_dimension); note that distance_to_boundaries is normalized by the particle radii.
        # clamp the distance to boundaries to be within [-rMax, rMax]
        normalized_distance_to_boundaries = torch.clamp(normalized_distance_to_boundaries, -self.rMax, self.rMax) # shape = (num_particles, 2*spatial_dimension)
        
        node_features.append(normalized_distance_to_boundaries)
        node_features.append(particle_radii) # shape = (num_particles, 1)
        """
        num_node_features:
            rotation = False: (C-1)*spatial_dimension + 2*spatial_dimension+ 1
            rotation = True: (C-1)*spatial_dimension + 2*spatial_dimension + 1 ...
        RF: interestingly, so as of now, this method should create node features with the same size as the input node features. There should be a better way to do this!
        """
        
        ### edge features
        edge_features = []
        receivers, senders = self.compute_graph_connectivity(current_position, num_particles_per_example)
        # normalized relative displacements across edges
        
        normalized_relative_displacements = (current_position[senders,:] - current_position[receivers,:])/(particle_radii[senders]+particle_radii[receivers]) # shape = (num_edges, spatial_dimension

        normalized_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True) # shape = (num_edges, 1)
        edge_features.append(torch.cat([normalized_relative_displacements, normalized_distances], dim=-1))

        normalized_relative_velocities = (last_velocities[senders] - last_velocities[receivers])/(particle_radii[senders]+particle_radii[receivers]).unsqueeze(-1) # shape = (num_edges, C-1, spatial_dimension)
        num_edges = len(receivers)
        flat_normalized_relative_velocities = normalized_relative_velocities.view(num_edges, -1) # shape = (num_edges, (C-1)*spatial_dimension)
        edge_features.append(flat_normalized_relative_velocities)
        edge_features.append(particle_radii[senders]+particle_radii[receivers]) # shape = (num_edges, 1)
        # normalized_absolute_relative_velocities = torch.norm(normalized_relative_velocities, dim=-1, keepdim=True) # shape = (num_edges, C-1, 1)
        # flat_normalized_absolute_relative_velocities = normalized_absolute_relative_velocities.view(num_edges, -1) # shape = (num_edges, (C-1))
        # edge_features.append(flat_normalized_absolute_relative_velocities)
        """
        num_edge_features:
            rotation = False: spatial_dimension + 1 + (C-1)*spatial_dimension
            rotation = True: spatial_dimension + 1 + (C-1)*spatial_dimension ...
        """

        # node_features is a list with element of shape (num_particles, N1), (num_particles, N2), ... . torch.cat(node_features, dim=-1) will concatenate these elements along the last dimension to get a shape of (num_particles, N1+N2+...). A similar scenario holds for edge_features.
        # torch.stack([senders, receivers]) will stack the tensors along the first dimension to get a shape of (2, num_edges).
        return torch.cat(node_features,dim=-1), torch.cat(edge_features, dim=-1), torch.stack([senders, receivers])

    def decoder_postprocessor(self,
                              acceleration: torch.tensor,
                              position_sequence: torch.tensor):
        """
        This internal method computes the predicted positions of the particles given the normalized accelerations and the current positions.
        Arguments:
            acceleration: tensor of shape (num_particles, spatial_dimension)
            position_sequence: tensor of shape (num_particles, C>1, spatial_dimension).
        
        Returns:
            predicted_positions: tensor of shape (num_particles, spatial_dimension)
        """
        # Use an Euler integrator to go from acceleration to position
        # RF: NEEDS TO BE FIXED; assumes dt = 1
        most_recent_position = position_sequence[:, -1, :]
        most_recent_velocity = position_sequence[:, -1, :] - position_sequence[:, -2, :]
        new_velocity = most_recent_velocity + acceleration
        new_position = most_recent_position + new_velocity
        return new_position

    def predict_position(self,
                         position_sequence: torch.tensor,
                         num_particles_per_example: torch.tensor,
                         particle_properties: torch.tensor):
        """
        This method predicts the positions of the particles given the current position sequence, the number of particles per example, and the particle properties. It performs a full pass of GNN, and calls the internal method decoder_postprocessor at the end to compute the predicted positions.
        Arguments:
            position_sequence: tensor of shape (num_particles, C, spatial_dimension)
            num_particles_per_example: tensor of shape (batch_size,) containing the number of particles in each example
            particle_properties: tensor of shape (num_particles, N) containing N properties of each particle
        
        Returns:
            predicted_position: tensor of shape (num_particles, spatial_dimension)
        """
        node_features, edge_features, edges = self.encoder_preprocessor(position_sequence, num_particles_per_example, particle_properties)
        predicted_normalized_acceleration = self.encoder_processor_decoder(node_features, edge_features, edges)
        particle_radii = self.get_particle_radii(particle_properties)
        predicted_position = self.decoder_postprocessor(predicted_normalized_acceleration*particle_radii, position_sequence)
        return predicted_position


    def predict_acceleration(self,
                             next_position: torch.tensor,
                             position_sequence: torch.tensor,
                             position_sequence_noise: torch.tensor,
                             num_particles_per_example: torch.tensor,
                             particle_properties: torch.tensor):
        """
        This method predicts the normalized accelerations given the next position (true training data), the current position sequence, and the noisy position sequence.
        Arguments:
            next_position: tensor of shape (num_particles, spatial_dimension)
            position_sequence: tensor of shape (num_particles, C, spatial_dimension)
            noisy_position_sequence: tensor of shape (num_particles, C, spatial_dimension)
            num_particles_per_example: tensor of shape (batch_size,) containing the number of particles in each example
            particle_properties: tensor of shape (num_particles, N) containing N properties of each particle
        Returns:
            predicted_normalized_acceleration: predicted normalized acceleration from the noisy position sequence; tensor of shape (num_particles, spatial_dimension)
            target_normalized_acceleration: predicted normalized acceleration while the velocity if computed noise free; tensor of shape (num_particles, spatial_dimension)
        """
        noisy_position_sequence = position_sequence + position_sequence_noise
        node_features, edge_features, edges = self.encoder_preprocessor(noisy_position_sequence, num_particles_per_example, particle_properties)
        predicted_normalized_acceleration = self.encoder_processor_decoder(node_features, edge_features, edges)
        next_position_adjusted = next_position + position_sequence_noise[:,-1,:] # ensures that the velocity is being computed noise free; acceleration will still be noisy however. An alternative is to let next_position_adjusted = next_positions + position_sequence_noise[:, -1] + (position_sequence_noise[:, -1] - position_sequence_noise[:, -2]), which ensures that the acceleration is noise free.
        particle_radii = self.get_particle_radii(particle_properties)
        target_normalized_acceleration = self.inverse_decoder_postprocessor(next_position_adjusted/particle_radii, noisy_position_sequence/particle_radii.unsqueeze(-1))

        return predicted_normalized_acceleration, target_normalized_acceleration

    def inverse_decoder_postprocessor(self,
                                      next_position_adjusted: torch.tensor,
                                      noisy_position_sequence: torch.tensor):
        """
        This internal method computes the target normalized acceleration given the adjusted next position and the noisy position sequence.
        """
        last_position = noisy_position_sequence[:, -1]
        last_velocity = last_position - noisy_position_sequence[:, -2] # this is the noisy velocity
        next_velocity = next_position_adjusted - last_position # this is the true velocity
        acceleration = next_velocity - last_velocity # wil include the noise due to last_velocity being noisy

        # acceleration_stats = self.normalization_stats["acc"]
        # normalized_acceleration = (acceleration - acceleration_stats['mean'])/acceleration_stats['std']
        return acceleration


    def save(self, path: str = 'model.pt'):
        torch.save(self.state_dict(), path)


    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


