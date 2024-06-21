import torch
import torch.nn as nn
import numpy as np
import network_architecture
import torch_geometric.nn as gnn
from typing import Dict, Tuple

# RF: what happened to dt?!
def time_diff(position_sequence: torch.tensor) -> torch.tensor:
    return position_sequence[:, 1:] - position_sequence[:, :-1]

class LearnedSimulator(nn.Module):
    """
    The constructor gets the following arguments:
    
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 num_message_passing_steps: int,
                 connectivity_radius: float,
                 normalization_stats: Dict['str', Dict['str', torch.tensor]],
                 boundaries: np.ndarray, # shape = (spatial_dimension, 2): lo and hi corners of the simulation box
                 rotation: bool = False,
                 spatial_dimension: int = 2,
                 num_encoded_node_features: int = 128,
                 num_encoded_edge_features: int = 128,
                 num_mlp_layers: int = 2,
                 mlp_layer_size: int = 128,
                 number_particle_types: int = 1,
                 particle_type_embedding_size: int = 16,
                 device="cpu"):
        super().__init__()
        self.connectivity_radius = connectivity_radius
        self.normalization_stats = normalization_stats
        print(f"normalization_stats: {normalization_stats}")
        self.boundaries = boundaries
        self.number_particle_types = number_particle_types

        self.particle_type_embedding = nn.Embedding(number_particle_types, particle_type_embedding_size)
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

    def forward(self):
        pass

    def compute_graph_connectivity(self,
                                   particle_locations: torch.tensor,
                                   num_particles_per_example: torch.tensor,
                                   add_self_edges: bool = False):
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
                             particle_types: torch.tensor,
                             material_property: torch.tensor = None):
        """
        This method encodes the particle positions and types using a sequence of C=6 most recent positions.

        Arguments:
            position_sequence: tensor of shape (num_particles, C, spatial_dimension)
            num_particles_per_example: tensor of shape (batch_size,) containing the number of particles in each example
            particle_types: tensor of shape (num_particles,) containing the type of each particle
            material_property: tensor of shape (num_particles,) containing the material property of each particle

        Returns:
            node_features: tensor of shape (num_particles, num_node_features)
            edge_features: tensor of shape (num_edges, num_edge_features)
            receivers, senders: tensors of shape (num_edges,) containing the receiver and sender indices of each edge
        """
        
        num_particles = position_sequence.shape[0]
        current_position = position_sequence[:, -1, :] # last position: shape = (num_particles, spatial_dimension)
        last_velocities = time_diff(position_sequence) # last C-1 velocities: shape = (num_particles, C-1, spatial_dimension)

        ### Encoded node features
        # flattened most recent velocities
        node_features = []
        velocity_stats = normalization_stats['vel']
        print(f"velocity_stats: {velocity_stats}")
        normalized_velocities = (last_velocities - velocity_stats["mean"])/velocity_stats["std"] # shape = (num_particles, C-1, spatial_dimension)
        flat_normalized_velocities = normalized_velocities.view(num_particles, -1) # shape = (num_particles, (C-1)*spatial_dimension)

        node_features.append(flat_normalized_velocities)

        # norlamized distance to boundaries
        boundaries = torch.tensor(self.boundaries, requires_grad=False).to(self.device) # converting boundaries from np.ndarray to torch.tensor; shape = (spatial_dimension, 2); boundaries[:,0] and boundaries[:,1] give the coordinates of the low and high corners of the simulation box with shape (spatial_dimension,), respectively.
        distannce_to_lower_boundary = current_position - boundaries[:, 0] # shape = (num_particles, spatial_dimension);
        distannce_to_upper_boundary = boundaries[:, 1] - current_position # shape = (num_particles, spatial_dimension);
        distance_to_boundaries = torch.cat([distannce_to_lower_boundary, distannce_to_upper_boundary], dim=-1)/self.connectivity_radius # shape = (num_particles, 2*spatial_dimension); note that distance_to_boundaries is normalized by the connectivity_radius.
        # # clip the distance to boundaries to [0,1], i.e., only consider distances that are less than or equal the connectivity_radius. Note that the distance_to_boundaries is always positive.
        # distance_to_boundaries = torch.clamp(distance_to_boundaries, max=1) # shape = (num_particles, 2*spatial_dimension)
        node_features.append(distance_to_boundaries)

        # particle types
        node_features.append(self.particle_type_embedding(particle_types))

        """
        num_node_features:
            rotation = False: (C-1)*spatial_dimension + 2*spatial_dimension + particle_type_embedding_size
            rotation = True: (C-1)*spatial_dimension + 2*spatial_dimension + particle_type_embedding_size + ...
        RF: interestingly, so as of now, this method should create node features with the same size as the input node features. There should be a better way to do this!
        """
        
        ### edge features
        edge_features = []
        receivers, senders = self.compute_graph_connectivity(current_position, num_particles_per_example)
        # normalized relative displacements across edges
        relative_displacements = (current_position[senders,:] - current_position[receivers,:])/self.connectivity_radius # shape = (num_edges, spatial_dimension)
        distances = torch.norm(relative_displacements, dim=-1, keepdim=True) # shape = (num_edges, 1)
        edge_features.append(torch.cat([relative_displacements, distances], dim=-1))
        """
        num_edge_features:
            rotation = False: spatial_dimension + 1
            rotation = True: spatial_dimension + 1 + ...
        """

        # node_features is a list with element of shape (num_particles, N1), (num_particles, N2), ... . torch.cat(node_features, dim=-1) will concatenate these elements along the last dimension to get a shape of (num_particles, N1+N2+...). A similar scenario holds for edge_features.
        # torch.stack([senders, receivers]) will stack the tensors along the first dimension to get a shape of (2, num_edges).
        return torch.cat(node_features,dim=-1), torch.stack([senders, receivers]), torch.cat(edge_features, dim=-1)

    
rotation = False
num_node_features = 30
num_edge_feartures = 3
num_encoded_node_features = 128
num_encoded_edge_features = 64
num_mlp_layers = 2
mlp_layer_size = 256
num_message_passing_steps = 5
connectivity_radius = 0.5
normalization_stats = {'vel': {'mean': torch.FloatTensor([0.01,0.0001]), 'std': torch.FloatTensor([0.005,0.00005])},
                       'acc': {'mean': torch.FloatTensor([0.05,0.0005]), 'std': torch.FloatTensor([0.002,0.00002])}}
boundaries = np.array([[0,0],[1,1]])


simulator = LearnedSimulator(num_node_features,
                             num_edge_feartures,
                             num_message_passing_steps,
                             connectivity_radius,
                             normalization_stats,
                             boundaries)


num_particles = 10
position_sequence = torch.rand(num_particles, 6, 2)
num_particles_per_example = torch.tensor([4,6])


print(simulator.output_node_size)
print(simulator.boundaries)
receivers, senders = simulator.compute_graph_connectivity(position_sequence[:,-1,:], num_particles_per_example)
print(f"senders: {senders}")
print(f"receivers: {receivers}")


particle_types = torch.full((num_particles,), 0)
node_features, edges, edge_features = simulator.encoder_preprocessor(position_sequence, num_particles_per_example, particle_types)
print(f"node_features: {node_features}")
print(f"node_features.shape: {node_features.shape}")
print(f"edges: {edges}")
print(f"edges.shape: {edges.shape}")
print(f"edge_features: {edge_features}")
print(f"edge_features.shape: {edge_features.shape}")
