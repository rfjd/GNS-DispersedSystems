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
                                   node_locations: torch.tensor,
                                   number_particles_per_example: torch:tensor):
        


rotation = False
num_node_features = 35
num_edge_feartures = 10
num_encoded_node_features = 128
num_encoded_edge_features = 64
num_mlp_layers = 2
mlp_layer_size = 256
num_message_passing_steps = 5
connectivity_radius = 0.1
normalization_stats = {'vel': (0.01,0.0001), 'acc': (0.05,0.0005)}
boundaries = np.array([[0,0],[1,1]])


simulator = LearnedSimulator(num_node_features,
                             num_edge_feartures,
                             num_message_passing_steps,
                             connectivity_radius,
                             normalization_stats,
                             boundaries)

print(simulator.output_node_size)
print(simulator.boundaries)
