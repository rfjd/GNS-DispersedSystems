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
NUM_NODE_FETURES = (C-1)*2+2*SPATIAL_DIMENSION+PARTICLE_TYPE_EMBEDDING_SIZE*(NUM_PARTICLE_TYPES>1) # e.g., C = 6: 5*2+2*2+8 = 30
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
position_sequence = torch.randn(num_particles, C, 2)
num_particles_per_example = torch.tensor([num_particles])
particle_types = torch.full((num_particles,), 0)


node_features, edge_features, edges = simulator.encoder_preprocessor(position_sequence, num_particles_per_example, particle_types)
print(f"node_features: {node_features}")
print(f"edge_features: {edge_features}")
print(f"edges: {edges}")

# CORRECT ANSWER:
# node_features: tensor([[-1.0124, -0.4075,  1.3856,  0.0991,  1.0000, -0.3141, -1.0000,  1.0000, -1.1258, -1.1524, -0.2506, -0.4339],
#                        [-1.5072, -0.4175, -0.5679,  0.1383, -1.0000, -1.0000,  1.0000,  1.0000, -1.1258, -1.1524, -0.2506, -0.4339],
#                        [ 0.5292,  0.2410, -0.2028,  0.1703, -1.0000, -0.7333,  1.0000,  1.0000, -1.1258, -1.1524, -0.2506, -0.4339]], grad_fn=<CatBackward0>)

# edge_features: tensor([[0.0000,  0.0000,  0.0000],[0.0000,  0.0000,  0.0000],[0.0902, -0.2669,  0.2817],[-0.0902,  0.2669,  0.2817],[0.0000,  0.0000,  0.0000]])
# edges = tensor([[0, 1, 1, 2, 2],[0, 1, 2, 1, 2]])
                       
predicted_position = simulator.predict_position(position_sequence, num_particles_per_example, particle_types) # tests encoder_preprocessor and encoder_processor_decoder
print(predicted_position)

# CORRECT ANSWER:
# tensor([[ 5.1671, -0.5728],
#         [ 0.2709, -0.7676],
#         [ 0.5818, -0.3499]], grad_fn=<AddBackward0>)


# train and rollout tests
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
# 3526.1171875.
# 2521.23046875.
# 2580.217041015625.
# 2324.1123046875.
# 1695.142578125.
# 3940.1767578125.
# 2953.656005859375.
# 2378.319580078125.
# 1977.849609375.
# 3629.54052734375.
# 2689.353759765625.

# Rollout Prediction Loss:
# 984.2992553710938
# 1040.2276611328125

## cleanup
os.system(f"rm {MODEL_PATH}model-* {MODEL_PATH}train_state-* {ROLLOUT_PATH}rollout_ex*")
