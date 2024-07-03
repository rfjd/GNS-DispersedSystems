#######################################################
"""
These tests are for develpmental purposes only. There are hardcoded values that are expected to be returned by the functions. These values are taken from an alternative gns implementation at https://github.com/geoelements/gns. Our code, only at its early stage mimics the functionality of that implementation. Actual tests can be found in tests.py.
"""
import sys
sys.path.append('../')

import torch
import numpy as np
from gns import seed_util
seed_util.initialize_seed(seed=0)

from gns import learned_simulator

C = 3 # input sequence length
NUM_ENCODED_NODE_FEATURES = 16
NUM_ENCODED_EDGE_FEATURES = 16
NUM_MLP_LAYERS = 2
MLP_LAYER_SIZE = 16
NUM_MESSAGE_PASSING_STEPS = 10
CONNECTIVITY_RADIUS = 1
SPATIAL_DIMENSION = 2
NUM_NODE_FETURES = (C-1)*2+2*SPATIAL_DIMENSION # e.g., C = 6: 5*2+2*2 = 14
NUM_EDGE_FEATURES = 3
normalization_stats = {'vel': {'mean': torch.FloatTensor([0.1,0.02]), 'std': torch.FloatTensor([1,4])},
                       'acc': {'mean': torch.FloatTensor([0.5,0.04]), 'std': torch.FloatTensor([2,3])}}

USE_PARTICLE_PROPERTIES = False
simulator = learned_simulator.LearnedSimulator(
    num_node_features=NUM_NODE_FETURES,
    num_edge_features=NUM_EDGE_FEATURES,
    num_message_passing_steps=NUM_MESSAGE_PASSING_STEPS,
    connectivity_radius=CONNECTIVITY_RADIUS,
    normalization_stats=normalization_stats,
    boundaries=np.array([[0,1],[0,1]]),
    num_encoded_node_features=NUM_ENCODED_NODE_FEATURES,
    num_encoded_edge_features=NUM_ENCODED_EDGE_FEATURES,
    num_mlp_layers=NUM_MLP_LAYERS,
    mlp_layer_size=MLP_LAYER_SIZE,
    device='cpu',
    use_particle_properties=USE_PARTICLE_PROPERTIES)

num_particles = 3
position_sequence = torch.rand(num_particles, C, 2)
num_particles_per_example = torch.tensor([num_particles])
particle_properties = torch.full((num_particles,), 0.5)

node_features, edge_features, edges = simulator.encoder_preprocessor(position_sequence, num_particles_per_example, particle_properties)
print(f"node_features: {node_features}")
print(f"edge_features: {edge_features}")
print(f"edges: {edges}")

# CORRECT ANSWER:
# node_features: tensor([
#     [ 0.1534,  0.1429,  0.2520, -0.0219,  0.9023,  0.7617,  0.0977,  0.2383],
#     [ 0.5998,  0.0581, -0.6701, -0.0936,  0.3242,  0.3931,  0.6758,  0.6069],
#     [-0.0687, -0.0890, -0.2022,  0.0354,  0.7832,  0.7705,  0.2168,  0.2295]])
# edge_features: tensor([
#     [ 0.0000,  0.0000,  0.0000],
#     [ 0.5782,  0.3686,  0.6857],
#     [ 0.1192, -0.0088,  0.1195],
#     [-0.5782, -0.3686,  0.6857],
#     [ 0.0000,  0.0000,  0.0000],
#     [-0.4590, -0.3774,  0.5942],
#     [-0.1192,  0.0088,  0.1195],
#     [ 0.4590,  0.3774,  0.5942],
#     [ 0.0000,  0.0000,  0.0000]])
# edges: tensor([
#     [0, 0, 0, 1, 1, 1, 2, 2, 2],
#     [0, 1, 2, 0, 1, 2, 0, 1, 2]])

predicted_position = simulator.predict_position(position_sequence, num_particles_per_example, particle_properties) # tests encoder_preprocessor and encoder_processor_decoder
print(predicted_position)

# CORRECT ANSWER:
# tensor([[1.9101, 2.0501],
#         [0.1099, 1.7092],
#         [1.3443, 2.2427]], grad_fn=<AddBackward0>)


#######################################################
# train and rollout tests
import sys
sys.path.append('../')

import torch
import numpy as np
from gns import seed_util
seed_util.initialize_seed(seed=0)

from gns import learned_simulator

import os
os.chdir("..")
DATA_PATH="test/sampleDATA/"
MODEL_PATH="test/sampleDATA/"
ROLLOUT_PATH="test/sampleDATA/"

number_steps=10

# os.system(f"mkdir -p {MODEL_PATH}")
# os.system(f"mkdir -p {ROLLOUT_PATH}")

C = 6 # input sequence length
NUM_ENCODED_NODE_FEATURES = 128
NUM_ENCODED_EDGE_FEATURES = 128
NUM_MLP_LAYERS = 2
MLP_LAYER_SIZE = 128
NUM_MESSAGE_PASSING_STEPS = 10
USE_PARTICLE_PROPERTIES = False

FLAGS = f"--C={C} --NUM_ENCODED_NODE_FEATURES={NUM_ENCODED_NODE_FEATURES} --NUM_ENCODED_EDGE_FEATURES={NUM_ENCODED_EDGE_FEATURES} --NUM_MLP_LAYERS={NUM_MLP_LAYERS} --MLP_LAYER_SIZE={MLP_LAYER_SIZE} --NUM_MESSAGE_PASSING_STEPS={NUM_MESSAGE_PASSING_STEPS} --USE_PARTICLE_PROPERTIES={USE_PARTICLE_PROPERTIES}"

# Train
os.system(f"python3 -m gns.main --data_path={DATA_PATH} --model_path={MODEL_PATH} --ntraining_steps={number_steps} " + FLAGS)

# Rollout Prediction
os.system(f"python3 -m gns.main --mode=rollout --data_path={DATA_PATH} --model_path={MODEL_PATH} --output_path={ROLLOUT_PATH} --model_file=model-{number_steps}.pt --train_state_file=train_state-{number_steps}.pt " + FLAGS)

# Expected output:

# Loss:
# 3517.70849609375
# 2532.505126953125.
# 2587.94091796875.
# 2318.3408203125.
# 1665.0546875.
# 3879.59716796875.
# 2926.023681640625.
# 2424.763427734375.
# 1957.2808837890625.
# 3620.72900390625.
# 2703.468994140625.

# Rollout Prediction Loss:
# 998.7435913085938
# 908.07470703125

## cleanup
os.system(f"rm {MODEL_PATH}model-* {MODEL_PATH}train_state-* {ROLLOUT_PATH}rollout_ex*")
