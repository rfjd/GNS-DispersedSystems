from gns import seed_util
seed_util.initialize_seed(seed=0)

import collections
import json
import os
import pickle
import glob
import re
import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from absl import flags
from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute

# Define the flags and their default values
flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.') # indicates how many of the examples are processed at once
flags.DEFINE_float('noise_std', 5e-4, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('testfilename', 'test', help='The name of the test file.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('output_filename', 'rollout', help='Base name for saving the rollout')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(50000), help='Number of steps at which to save the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
# flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')
flags.DEFINE_integer('lr_decay_steps', int(2e5), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
flags.DEFINE_integer("n_gpus", 1, help="The number of GPUs to utilize for training")

FLAGS = flags.FLAGS

# global variables
flags.DEFINE_boolean('ROTATION', False, 'Whether to use rotation or not')
flags.DEFINE_integer('C', 6, 'Input sequence length')
flags.DEFINE_integer('NUM_ENCODED_NODE_FEATURES', 128, 'Number of encoded node features')
flags.DEFINE_integer('NUM_ENCODED_EDGE_FEATURES', 128, 'Number of encoded edge features')
flags.DEFINE_integer('NUM_MLP_LAYERS', 2, 'Number of MLP layers')
flags.DEFINE_integer('MLP_LAYER_SIZE', 128, 'Size of each MLP layer')
flags.DEFINE_integer('NUM_MESSAGE_PASSING_STEPS', 10, 'Number of message passing steps')
flags.DEFINE_integer('SPATIAL_DIMENSION', 2, 'Spatial dimension')

FLAGS(sys.argv)
C = FLAGS.C
NUM_ENCODED_NODE_FEATURES = FLAGS.NUM_ENCODED_NODE_FEATURES
NUM_ENCODED_EDGE_FEATURES = FLAGS.NUM_ENCODED_EDGE_FEATURES
NUM_MLP_LAYERS = FLAGS.NUM_MLP_LAYERS
MLP_LAYER_SIZE = FLAGS.MLP_LAYER_SIZE
NUM_MESSAGE_PASSING_STEPS = FLAGS.NUM_MESSAGE_PASSING_STEPS
SPATIAL_DIMENSION = FLAGS.SPATIAL_DIMENSION
NUM_NODE_FEATURES = (C-1)*2+2*SPATIAL_DIMENSION+1 # e.g., C = 6: 5*2+2*2+1 = 15
NUM_EDGE_FEATURES = (C-1)*2+3+1 # e.g., C = 6: 5*2+3+1 = 14

def rollout(simulator: learned_simulator.LearnedSimulator,
            position_sequence: torch.tensor,
            particle_properties: torch.tensor,
            num_particles_per_example: torch.tensor,
            num_steps: int,
            device: torch.device):
    """
    Generate a rollout using the first C steps of the position_sequence
    Argumentss:
        simulator (learned_simulator.LearnedSimulator): simulator object
        position_sequence (torch.Tensor): positions of the particles; shape = (num_particles, sequence_length, 2). Here, sequence_length is the total number of time steps of the simulation.
        num_steps (int): sequence length - C
    
    Returns:
        rollout_dict (dict): dictionary containing the initial sequence, predicted and ground truth rollouts, and particle types
        loss: squared error loss between the predicted and ground truth rollouts
    """
    ground_truth_positions = position_sequence[:, C:, :]

    current_position_sequence = position_sequence[:, :C, :]
    predictions = []
    for step in range(num_steps):
        predicted_position = simulator.predict_position(current_position_sequence,
                                                        [num_particles_per_example],
                                                        particle_properties) # shape = (num_particles, spatial_dimension)
        predictions.append(predicted_position)
        current_position_sequence = torch.cat([current_position_sequence[:, 1:, :], predicted_position[:,None,:]], dim=1) # shift the sequence forward by one step; note that the predicted new position is added to the sequence, and not the corresponding ground truth position.

    predictions = torch.stack(predictions, dim=0) # shape = (num_steps, num_particles, spatial_dimension)

    ground_truth_positions = ground_truth_positions.permute(1,0,2)# shape = (num_steps, num_particles, spatial_dimension) RF: why permute the ground truvh instead of the predictions?
    loss = (ground_truth_positions-predictions)**2

    rollout_dict = {
        'initial_positions': position_sequence[:, :C, :].permute(1, 0, 2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_properties': particle_properties.cpu().numpy()
    }

    return rollout_dict, loss


def get_simulator(metadata: json,
                  acc_noise_std: float,
                  vel_noise_std: float,
                  device: torch.device):
    """
    Instantiates the learned simulator.

    Args:
      metadata: JSON object with metadata.
      acc_noise_std: Acceleration noise std deviation.
      vel_noise_std: Velocity noise std deviation.
      device: PyTorch device 'cpu' or 'cuda'.

    Returns:
      simulator: LearnedSimulator object
    """

    # RF why do we need to add noise to the std deviation?
    # normalization_stats = {
    #     'acc': {'mean': torch.FloatTensor(metadata['acc_mean']).to(device),'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 + acc_noise_std**2).to(device)},
    #     'vel': {'mean': torch.FloatTensor(metadata['vel_mean']).to(device),'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2+vel_noise_std**2).to(device)}
    # }

    simulator = learned_simulator.LearnedSimulator(
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        num_message_passing_steps=NUM_MESSAGE_PASSING_STEPS,
        connectivity_radius=metadata['default_connectivity_radius'],
        # normalization_stats=normalization_stats,
        boundaries=np.array(metadata['bounds']),
        num_encoded_node_features=NUM_ENCODED_NODE_FEATURES,
        num_encoded_edge_features=NUM_ENCODED_EDGE_FEATURES,
        num_mlp_layers=NUM_MLP_LAYERS,
        mlp_layer_size=MLP_LAYER_SIZE,
        device=device)

    return simulator


def predict(device: str):
    """
    This function loads a learned simulator (trained model) and generates a rollout on validation or test data.
    Arguments:
        device (str): device to run the simulation on ('cpu' or 'cuda')
    """
    file = open(f"{FLAGS.output_path}/rollout_loss.txt", 'w')
    metadata = reading_utils.read_metadata(FLAGS.data_path)
    simulator = get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)
    simulator.load(FLAGS.model_path + FLAGS.model_file) # load the pre-trained model; map_location='cpu'
    simulator.to(device)
    simulator.eval() # set the model to evaluation mode

    # Output path
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    # Load the dataset
    ds = data_loader.get_data_loader_by_trajectories(f"{FLAGS.data_path}{FLAGS.testfilename}.npz") # list of trajectrory examples with content (positions, particle_properties, num_particles_in_example)
    eval_loss = []
    with torch.no_grad():
        for example_id, trajectory_data in enumerate(ds):
            position_sequence = trajectory_data[0].to(device)
            particle_properties = trajectory_data[1].to(device) 
            num_particles_in_example = torch.tensor([int(trajectory_data[2])], dtype=torch.int32).to(device)
            sequence_length = position_sequence.shape[1]
            num_steps = sequence_length - C

            rollout_dict, loss = rollout(simulator,
                                         position_sequence,
                                         particle_properties,
                                         num_particles_in_example,
                                         num_steps,
                                         device)
            
            eval_loss.append(torch.flatten(loss))
            
            rollout_dict['metadata'] = metadata
            rollout_dict['loss'] = loss.mean()
            print(f"Predicting example {example_id} with loss {rollout_dict['loss']}")
            filename = f'{FLAGS.output_filename}_ex{example_id}.pkl'
            filename = os.path.join(FLAGS.output_path, filename)
            with open(filename, 'wb') as f:
                pickle.dump(rollout_dict, f)

            file.write(f"{rollout_dict['loss']}\n")

    file.close()
    print(f"Mean loss on rollout prediction: {torch.mean(torch.cat(eval_loss))}")

#### RF: to be modified
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def train(rank, flags, world_size, device):
    file = open(f"{flags['model_path']}/loss.txt", 'w')
    """Train the model.

    Args:
        rank: local rank
        world_size: total number of ranks
        device: torch device type
    """
    if device == torch.device("cuda"):
        distribute.setup(rank, world_size, device)
        device_id = rank
    else:
        device_id = device

    # Read metadata
    metadata = reading_utils.read_metadata(flags["data_path"])

    # Get simulator and optimizer
    if device == torch.device("cuda"):
        serial_simulator = get_simulator(metadata, flags["noise_std"], flags["noise_std"], rank)
        simulator = DDP(serial_simulator.to(rank), device_ids=[rank], output_device=rank)
        optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*world_size)
    else:
        simulator = get_simulator(metadata, flags["noise_std"], flags["noise_std"], device)
        optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"] * world_size)
    step = 0
    
    # print(f"simulator is {simulator}")
    # If model_path does exist and model_file and train_state_file exist continue training.
    if flags["model_file"] is not None:

        if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(f'{flags["model_path"]}*model*pt')
            max_model_number = 0
            expr = re.compile(".*model-(\d+).pt")
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            # reset names to point to the latest.
            flags["model_file"] = f"model-{max_model_number}.pt"
            flags["train_state_file"] = f"train_state-{max_model_number}.pt"

        if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
            # load model
            if device == torch.device("cuda"):
                simulator.module.load(flags["model_path"] + flags["model_file"])
            else:
                simulator.load(flags["model_path"] + flags["model_file"])

            # load train state
            train_state = torch.load(flags["model_path"] + flags["train_state_file"])
            # set optimizer state
            optimizer = torch.optim.Adam(
                simulator.module.parameters() if device == torch.device("cuda") else simulator.parameters())
            optimizer.load_state_dict(train_state["optimizer_state"])
            optimizer_to(optimizer, device_id)
            # set global train state
            step = train_state["global_train_state"].pop("step")

        else:
            msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
            raise FileNotFoundError(msg)

    simulator.train()
    simulator.to(device_id)

    if device == torch.device("cuda"):
        dl = distribute.get_data_distributed_dataloader_by_samples(path=f'{flags["data_path"]}train.npz', input_length_sequence=C, batch_size=flags["batch_size"])
    else:
        dl = data_loader.get_data_loader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                    input_length_sequence=C,
                                                    batch_size=flags["batch_size"])

    # print(f"len(dl.dataset._data) is {len(dl.dataset._data)}") # = 6, number of simulations (examples) in the dataset; dl: data loader; dal.dataset: SamplesDataset
    # print(f"dl.dataset._data[0].shape is {dl.dataset._data[0].shape}")
    # print(f"dl.dataset._data[0][0].shape is {dl.dataset._data[0][0].shape}")
    # print(f"n_features is {len(dl.dataset._data[0])}")
    n_features = len(dl.dataset._data[0]) # Horrible naming! This the size of the Tuple; if =2, it means only particle positions and material type are given. If 3, material property is also given.

    COUNTER = 0
    print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")
    not_reached_nsteps = True
    try:
        while not_reached_nsteps:
            if device == torch.device("cuda"):
                torch.distributed.barrier()
            else:
                pass
            for example in dl:  # ((position, particle_properties, material_property, n_particles_per_example), labels) are in dl
                COUNTER += 1
                # print(f"counter is {COUNTER}")
                # print(f"inside for example in dl loop") # example here is a list; example[i] is also a list
                # print(f"len(example) is {len(example)}")
                # print(f"example[0][0].shape is {example[0][0].shape}") # (num_particles, 6, DIM) last 6 positions; here number_particles is the total number of particles in batch_size examples
                # print(f"example[0][1].shape is {example[0][1].shape}") # (num_particles, ) particle properties
                # print(f"example[1].shape is {example[1].shape}")
                # print(f"example[1] is {example[1]}") # What is this? The second entry of the example list/tuple
                position = example[0][0].to(device_id)
                particle_properties = example[0][1].to(device_id)
                if n_features == 3:  # if dl includes material_property
                    material_property = example[0][2].to(device_id)
                    n_particles_per_example = example[0][3].to(device_id)
                elif n_features == 2:
                    n_particles_per_example = example[0][2].to(device_id)
                else:
                    raise NotImplementedError
                labels = example[1].to(device_id)

                n_particles_per_example.to(device_id)
                labels.to(device_id)

                # Sample the noise to add to the inputs to the model during training.
                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=flags["noise_std"]).to(device_id)
                # Get the predictions and target accelerations.
                if device == torch.device("cuda"):
                    # print(f"inside the train loop for cuda block")
                    # print(f"nparticles_per_example is {n_particles_per_example}")
                    pred_acc, target_acc = simulator.module.predict_acceleration(
                        next_position=labels.to(rank),
                        position_sequence=position.to(rank),
                        position_sequence_noise=sampled_noise.to(rank),
                        num_particles_per_example=n_particles_per_example.to(rank),
                        particle_properties=particle_properties.to(rank)
                    )
                else:
                    pred_acc, target_acc = simulator.predict_acceleration(
                        next_position=labels.to(device),
                        position_sequence=position.to(device),
                        position_sequence_noise=sampled_noise.to(device),
                        num_particles_per_example=n_particles_per_example.to(device),
                        particle_properties=particle_properties.to(device)
                    )

                # Calculate the loss and take an average
                loss = (pred_acc - target_acc)**2
                loss = loss.sum()/len(loss)

                # Computes the gradient of loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                # exponential decay
                lr_new = flags["lr_init"]*(flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
                # # step decay
                # lr_new = flags["lr_init"]*(flags["lr_decay"]**(step//flags["lr_decay_steps"])) * world_size
                for param in optimizer.param_groups:
                    param['lr'] = lr_new

                if rank == 0 or device == torch.device("cpu"):
                    line = f'Training step: {step}/{flags["ntraining_steps"]}. Loss: {loss}.' 
                    print(line, flush=True)
                    file.write(f"{loss}\n")
                    # Save model state
                    if step % flags["nsave_steps"] == 0:
                        if device == torch.device("cpu"):
                            simulator.save(flags["model_path"] + 'model-'+str(step)+'.pt')
                        else:
                            simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
                        train_state = dict(optimizer_state=optimizer.state_dict(),
                                           global_train_state={"step": step},
                                           loss=loss.item())
                        torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

                # Complete training
                if (step >= flags["ntraining_steps"]):
                    not_reached_nsteps = False
                    break

                step += 1

    except KeyboardInterrupt:
        pass

    if rank == 0 or device == torch.device("cpu"):
        if device == torch.device("cpu"):
            simulator.save(flags["model_path"] + 'model-'+str(step)+'.pt')
        else:
            simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
        train_state = dict(optimizer_state=optimizer.state_dict(),
                           global_train_state={"step": step},
                           loss=loss.item())
        torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

    if torch.cuda.is_available():
        distribute.cleanup()

    file.close()

def main(_):
    """Train or evaluates the model."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    myflags = reading_utils.flags_to_dict(FLAGS)

    if FLAGS.mode == 'train':
        # If model_path does not exist create new directory.
        if not os.path.exists(FLAGS.model_path):
            os.makedirs(FLAGS.model_path)

        # Train on GPU
        if device == torch.device('cuda'):
            available_gpus = torch.cuda.device_count()
            print(f"Available GPUs = {available_gpus}")

            # Set the number of GPUs based on availability and the specified number
            if FLAGS.n_gpus is None or FLAGS.n_gpus > available_gpus:
                world_size = available_gpus
                if FLAGS.n_gpus is not None:
                    print(f"Warning: The number of GPUs specified ({FLAGS.n_gpus}) exceeds the available GPUs ({available_gpus})")
            else:
                world_size = FLAGS.n_gpus

            # Print the status of GPU usage
            print(f"Using {world_size}/{available_gpus} GPUs")

            # Spawn training to GPUs
            distribute.spawn_train(train, myflags, world_size, device)

        # Train on CPU 
        else:
            rank = None
            world_size = 1
            train(rank, myflags, world_size, device)

    elif FLAGS.mode in ['valid', 'rollout']:
        # Set device
        world_size = torch.cuda.device_count()
        if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
            device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
        # Test code
        print(f"device is {device} world size is {world_size}")
        predict(device)

if __name__ == '__main__':
  app.run(main)
