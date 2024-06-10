from typing import List
import torch
import torch.nn as nn


"""
This function builds a multilayer perceptron (MLP) for nodes and edges features.

Arguments:
    input_size: size of the input to MLP (e.g., number of features for a node or edge)
    hidden_layer_sizes: a list of the number of neurons in each hidden layer
    output_size: size of the output of the MLP (i.e., embedding of the node or edge features)
    hidden_activation: activation of the hidden layers
    output_activation: activation of the output layer

Returns:
    mlp: an MLP sequential cotainer
"""
def build_mlp(input_size: int,
              hidden_layer_sizes: List[int],
              output_size: int,
              hidden_activation: nn.Module = nn.ReLU,
              output_activation: nn.Module = nn.Identity) -> nn.Sequential:

    # create a list of activations
    num_hidden_layers = len(hidden_layer_sizes)
    layer_activations = [hidden_activation for _ in range(num_hidden_layers)]
    layer_activations.append(output_activation)

    # a list of number of neurons in each layer
    layer_sizes = [input_size,*hidden_layer_sizes,output_size]
    num_active_layers = len(layer_sizes)-1 # excluding the input layer
    
    # Create the sequential container
    # For each hidden layer and the output layer, add a linear transformation followed by an activation function
    mlp = nn.Sequential()
    for i in range(num_active_layers):
        mlp.add_module(f"Linear-{i}", nn.Linear(layer_sizes[i],layer_sizes[i+1])) # a linear transformation layer
        mlp.add_module(f"Activation-{i}", layer_activations[i]()) # activation

    return mlp



"""
This class encodes the node and edge features using MLP. The constructor the following arguments:
    num_node_features: number of features for nodes
    node_embedding_size: size of the node embedding by MLP
    num_edge_features: number of features for edges
    edge_embedding_size: size of the edge embedding by MLP
    num_mlp_layers: number of hidden layers in MLP
    mlp_layer_size: number of neurons in the MLP hidden layers
"""
class Encoder(nn.Module):
    def __init__(self,
                 num_node_features: int,
                 node_embedding_size: int,
                 num_edge_features: int,
                 edge_embedding_size: int,
                 num_mlp_layers: int,
                 mlp_layer_size):
        super().__init__()
        mlp_node = build_mlp(num_node_features,
                             [mlp_layer_size for _ in range(num_mlp_layers)],
                             node_embedding_size)
        self.node_encode = nn.Sequential(*mlp_node,nn.LayerNorm(node_embedding_size))

        mlp_edge = build_mlp(num_edge_features,
                             [mlp_layer_size for _ in range(num_mlp_layers)],
                             edge_embedding_size)
        self.edge_encode = nn.Sequential(*mlp_edge,nn.LayerNorm(edge_embedding_size))


    """
    The forward method below encodes the nodes and edges.

    Arguments:
        x: node features; shape = (number_particles, num_node_features)
        e: edge features; shape = (number_particles, num_edge_features)

    Returns:
        node_encode(x): encoded node features
        edge_encode(e): encoded edge features
    """
    def forward(self, x: torch.tensor, e: torch.tensor):
        return self.node_encode(x), self.edge_encode(e)


# num_node_features = 10
# node_embedding_size = 13
# num_edge_features = 15
# edge_embedding_size = 7
# mlp_layer_size = 128
# num_mlp_layers = 2

# mlp = build_mlp(num_node_features,
#                 [mlp_layer_size for _ in range(num_mlp_layers)],
#                 node_embedding_size)
# print(nn.Sequential(*mlp_node,nn.LayerNorm(node_embedding_size)))

# encoder = Encoder(num_node_features, node_embedding_size, num_edge_features, edge_embedding_size, num_mlp_layers, mlp_layer_size)

# number_particles = 2;
# x = torch.rand(number_particles, num_node_features)
# e = torch.rand(number_particles, num_edge_features)
# encoded_x, encoded_e = encoder.forward(x,e)
# print(encoded_x)
# print(encoded_e)
