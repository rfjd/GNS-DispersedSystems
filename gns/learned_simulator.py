import torch
import torch.nn as nn
import numpy as np
import network_architecture
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
                 normalization_stats: Dict['str', Tuple[float, float]],
                 boundaries: np.ndarray,
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


rotation = False
num_node_features = 35
num_edge_feartures = 10
num_encoded_node_features = 128
num_encoded_edge_features = 64
num_mlp_layers = 2
mlp_layer_size = 256
num_message_passing_steps = 5
connectivity_radius = 0.5
normalization_stats = {'vel': (0.01,0.0001), 'acc': (0.05,0.0005)}
boundaries = np.array([[0,0],[1,1]])


simulator = LearnedSimulator(num_node_features,
                             num_edge_feartures,
                             num_message_passing_steps,
                             connectivity_radius,
                             normalization_stats,
                             boundaries)


num_particles = 10
particle_locations = torch.rand(num_particles,2)
num_particles_per_example = torch.tensor([4,6])


print(simulator.output_node_size)
print(simulator.boundaries)
receivers, senders = simulator.compute_graph_connectivity(particle_locations, num_particles_per_example)
print(f"receivers: {receivers}")
print(f"senders: {senders}")
