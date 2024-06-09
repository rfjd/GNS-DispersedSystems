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
              output_activation: nn.Module = nn.Identity):# -> nn.Sequential:

    # create a list of activations
    num_hidden_layers = len(hidden_layer_sizes)
    layer_activations = [hidden_activation for i in range(num_hidden_layers)]
    layer_activations.append(output_activation)

    # a list of number of neurons in each layer
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    num_active_layers = len(layer_sizes)-1 # excluding the input layer
    
    # Create the sequential container
    # For each hidden layer and the output layer, add a linear transformation followed by an activation function
    mlp = nn.Sequential()
    for i in range(num_active_layers):
        mlp.add_module(f"Linear-{i}", nn.Linear(layer_sizes[i],layer_sizes[i+1])) # a linear transformation layer
        mlp.add_module(f"Activation-{i}", layer_activations[i]()) # activation

    return mlp


print(build_mlp(10,[3,4,5],11))
