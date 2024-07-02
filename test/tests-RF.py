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
NUM_PARTICLE_TYPES = 9
PARTICLE_TYPE_EMBEDDING_SIZE = 4
SPATIAL_DIMENSION = 2
NUM_NODE_FETURES = (C-1)*2+2*SPATIAL_DIMENSION #(C-1)*2+2*SPATIAL_DIMENSION+PARTICLE_TYPE_EMBEDDING_SIZE*(NUM_PARTICLE_TYPES>1) # e.g., C = 6: 5*2+2*2+8 = 30
NUM_EDGE_FEATURES = 3
normalization_stats = {'vel': {'mean': torch.FloatTensor([0.1,0.02]), 'std': torch.FloatTensor([1,4])},
                       'acc': {'mean': torch.FloatTensor([0.5,0.04]), 'std': torch.FloatTensor([2,3])}}

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
        number_particle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=PARTICLE_TYPE_EMBEDDING_SIZE,
        device='cpu')

num_particles = 3
position_sequence = torch.rand(num_particles, C, 2)
num_particles_per_example = torch.tensor([num_particles])
particle_types = torch.full((num_particles,), 0)

node_features, edge_features, edges = simulator.encoder_preprocessor(position_sequence, num_particles_per_example, particle_types)
print(f"node_features: {node_features}")
print(f"edge_features: {edge_features}")
print(f"edges: {edges}")

# CORRECT ANSWER:
# node_features: tensor([
#         [-0.3025,  0.1241, -0.1016, -0.0535,  0.1457,  0.7009,  0.8543,  0.2991],
#         [ 0.2927, -0.0116, -0.5408,  0.1663,  0.3968,  0.8576,  0.6032,  0.1424],
#         [-0.2223, -0.0584, -0.7056, -0.0986,  0.1002,  0.3788,  0.8998,  0.6212]])
# edge_features: tensor([[ 0.0000,  0.0000,  0.0000],
#         [-0.2512, -0.1567,  0.2960],
#         [ 0.0455,  0.3220,  0.3252],
#         [ 0.2512,  0.1567,  0.2960],
#         [ 0.0000,  0.0000,  0.0000],
#         [ 0.2966,  0.4787,  0.5632],
#         [-0.0455, -0.3220,  0.3252],
#         [-0.2966, -0.4787,  0.5632],
#         [ 0.0000,  0.0000,  0.0000]])
# edges: tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
#         [0, 1, 2, 0, 1, 2, 0, 1, 2]])

predicted_position = simulator.predict_position(position_sequence, num_particles_per_example, particle_types) # tests encoder_preprocessor and encoder_processor_decoder
print(predicted_position)

# CORRECT ANSWER:
# tensor([[ 0.4564,  0.4120],
#         [ 0.2455,  1.5968],
#         [-0.1927, -0.1386]], grad_fn=<AddBackward0>)


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
NUM_PARTICLE_TYPES = 9
PARTICLE_TYPE_EMBEDDING_SIZE = 16

FLAGS = f"--C={C} --NUM_ENCODED_NODE_FEATURES={NUM_ENCODED_NODE_FEATURES} --NUM_ENCODED_EDGE_FEATURES={NUM_ENCODED_EDGE_FEATURES} --NUM_MLP_LAYERS={NUM_MLP_LAYERS} --MLP_LAYER_SIZE={MLP_LAYER_SIZE} --NUM_MESSAGE_PASSING_STEPS={NUM_MESSAGE_PASSING_STEPS} --NUM_PARTICLE_TYPES={NUM_PARTICLE_TYPES} --PARTICLE_TYPE_EMBEDDING_SIZE={PARTICLE_TYPE_EMBEDDING_SIZE}"

# Train
os.system(f"python3 -m gns.main --data_path={DATA_PATH} --model_path={MODEL_PATH} --ntraining_steps={number_steps} " + FLAGS)

# Rollout Prediction
os.system(f"python3 -m gns.main --mode=rollout --data_path={DATA_PATH} --model_path={MODEL_PATH} --output_path={ROLLOUT_PATH} --model_file=model-{number_steps}.pt --train_state_file=train_state-{number_steps}.pt " + FLAGS)

# Expected output:

# Loss:
# 3503.91357421875
# 2547.35107421875
# 2566.610107421875
# Loss: 2321.4375
# 1680.4036865234375
# 3924.21044921875
# 2944.2734375
# 2414.891845703125
# 1973.1336669921875
# 3627.047607421875
# 2706.091552734375

# Rollout Prediction Loss:
# 1251.09130859375
# 1117.198974609375

## cleanup
os.system(f"rm {MODEL_PATH}model-* {MODEL_PATH}train_state-* {ROLLOUT_PATH}rollout_ex*")
