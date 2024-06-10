from typing import List
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

def build_mlp(input_size: int,
              hidden_layer_sizes: List[int],
              output_size: int,
              hidden_activation: nn.Module = nn.ReLU,
              output_activation: nn.Module = nn.Identity) -> nn.Sequential:

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



class Encoder(nn.Module):
    """
    This class encodes the node and edge features using MLP. The constructor gets the following arguments:
        num_node_features: number of features for nodes
        node_embedding_size: size of the node embedding by MLP
        num_edge_features: number of features for edges
        edge_embedding_size: size of the edge embedding by MLP
        num_mlp_layers: number of hidden layers in MLP
        mlp_layer_size: number of neurons in the MLP hidden layers
    """
    def __init__(self,
                 num_node_features: int,
                 node_embedding_size: int,
                 num_edge_features: int,
                 edge_embedding_size: int,
                 num_mlp_layers: int,
                 mlp_layer_size):
        super().__init__()
        node_mlp = build_mlp(num_node_features,
                             [mlp_layer_size for _ in range(num_mlp_layers)],
                             node_embedding_size)
        self.node_mlp = nn.Sequential(*node_mlp,nn.LayerNorm(node_embedding_size))

        edge_mlp = build_mlp(num_edge_features,
                             [mlp_layer_size for _ in range(num_mlp_layers)],
                             edge_embedding_size)
        self.edge_mlp = nn.Sequential(*edge_mlp,nn.LayerNorm(edge_embedding_size))


    def forward(self, x: torch.tensor, e: torch.tensor):
        """
        The forward method below encodes the nodes and edges.

        Arguments:
            x: node features; shape = (number_particles, num_node_features)
            e: edge features; shape = (number_edges, num_edge_features) [RF: TOBECHECKED]

        Returns:
            node_encode(x): encoded node features
            edge_encode(e): encoded edge features
        """
        return self.node_mlp(x), self.edge_mlp(e)




# RF: the gns-Pytorch code seems to have a few mistakes in this class. First, since we stack mutiple message passings in the Processor, the number of features in and out of the MLP should be the same for both nodes and edges. Second, the node_mlp method has a mistake in the number of input features; what's in the code (nnode_in + nedge_out) should change to 2*{number of node features} (in or out since they should be equal).
class InteractionNetwork(gnn.MessagePassing):
    """
    This class handles message passing and is the building block of the Processor class. The constructor gets the following arguments:
        num_encoded_node_features: number of node features
        num_encoded_edge_features: number of edge features
        num_mlp_layers: number of hidden layers in MLP
        mlp_layer_size: number of neurons in the MLP hidden layers
    """
    def __init__(self,
                 num_encoded_node_features: int,
                 num_encoded_edge_features: int,
                 num_mlp_layers: int,
                 mlp_layer_size: int):
        super().__init__(aggr='add')
        # method for node MLP
        # features a node and its updated version go through MLP; see the update method below.
        node_mlp = build_mlp(2*num_encoded_node_features,
                             [mlp_layer_size for _ in range(num_mlp_layers)],
                             num_encoded_node_features)
        self.node_mlp = nn.Sequential(*node_mlp,nn.LayerNorm(num_encoded_node_features))

        # features from two nodes and the connecting edge between them go through MLP; see the message method below.
        edge_mlp = build_mlp(num_encoded_edge_features+2*num_encoded_node_features,
                             [mlp_layer_size for _ in range(num_mlp_layers)],
                             num_encoded_edge_features)
        self.edge_mlp = nn.Sequential(*edge_mlp,nn.LayerNorm(num_encoded_edge_features))


    def forward(self,
                x: torch.tensor,
                e: torch.tensor,
                edge_index: torch.tensor):
        """
        The forward method passes messages and adds residuals to the updated embeddings.

        Arguments:
            x: node embeddings; shape = (number_particles, num_encoded_node_features)
            e: edge embeddings; shape = (number_edges, num_encoded_edge_features)
            edge_index: tensor with shape (2, number_edges) indicating connection between nodes (first row: source, second row: target)

        Returns: updated node and edge embeddings0
        """
        x_residual, e_residual = x, e
        self.propagate(edge_index, x=x, edge_features=e)
        return x+x_residual, e+e_residual


# num_node_features = 10
# node_embedding_size = 13
# num_edge_features = 15
# edge_embedding_size = 7
# mlp_layer_size = 128
# num_mlp_layers = 2

# node_mlp = build_mlp(num_node_features,
#                 [mlp_layer_size for _ in range(num_mlp_layers)],
#                 node_embedding_size)
# print(nn.Sequential(*node_mlp,nn.LayerNorm(node_embedding_size)))

# encoder = Encoder(num_node_features, node_embedding_size, num_edge_features, edge_embedding_size, num_mlp_layers, mlp_layer_size)

# number_particles = 3
# number_edges = 2
# x = torch.rand(number_particles, num_node_features)
# e = torch.rand(number_edges, num_edge_features)
# encoded_x, encoded_e = encoder.forward(x,e)
# print(encoded_x)
# print(encoded_e)

torch.manual_seed(42)
number_particles = 3
number_edges = 3
number_embedded_node_features = 12
number_embedded_edge_features = 7
num_mlp_layers = 2
mlp_layer_size = 74
GN = InteractionNetwork(number_embedded_node_features,number_embedded_edge_features,num_mlp_layers,mlp_layer_size)
# x = torch.rand(number_particles, number_embedded_node_features)
# e = torch.rand(number_edges, number_embedded_edge_features)
x=torch.tensor([[4.4647e-01, 6.0369e-01, 5.7371e-01, 3.4733e-01, 7.6309e-01, 6.6871e-01,
         1.7654e-02, 3.6207e-01, 2.0304e-01, 7.1463e-01, 7.7116e-04, 7.0299e-02],
        [5.9259e-02, 3.2826e-01, 1.1441e-01, 4.0033e-01, 2.1547e-01, 8.2144e-01,
         9.3869e-01, 6.9342e-01, 3.0600e-01, 7.0400e-01, 7.0669e-02, 2.4992e-01],
        [3.9518e-01, 3.2514e-02, 9.6887e-01, 8.2764e-01, 7.3163e-01, 7.3500e-01,
         9.9204e-01, 3.0989e-01, 6.1994e-01, 3.3186e-01, 4.1512e-01, 9.0099e-02]])
e=torch.tensor([[0.7990, 0.1398, 0.2765, 0.9457, 0.2066, 0.5643, 0.0411],
        [0.2096, 0.9536, 0.7689, 0.5212, 0.2507, 0.8813, 0.9706],
        [0.6002, 0.1777, 0.5172, 0.7436, 0.6460, 0.9499, 0.9503]])

edge_index = torch.tensor([[0, 0, 1],
                           [1, 2, 2]])

print(edge_index.shape)
print(f"x={x}")
print(f"e={e}")
print("Message Passing...")
x, e = GN.forward(x, e, edge_index)
print(f"x={x}")
print(f"e={e}")
