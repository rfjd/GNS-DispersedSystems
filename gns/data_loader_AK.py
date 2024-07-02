from gns import seed_util

seed_util.apply_seed()

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from intervaltree import IntervalTree


def load_npz_file(file_path):
    """
    Load data stored in npz format.
    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.
    Args:
        file_path (str): Path to npz file.
    Returns:
        data (list): List of tuples of the form (positions, particle_type).
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        return [item for _, item in data.items()]
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")


class TrajectoriesSampleDataset(Dataset):
    """
    Trajectories dataset where each trajectory is a series of tuples
        (positions, particle_type, material_property[optional]).
    positions' shape: (sequence_length, n_particle, dimension)
        where dimension is the number of coordinates (e.g. 2 for x and y)
    particle type shape: (n_particles)
    [optional] material_property: (n_particles, feature vector)

    This dataset returns a sequence of a trajectory with "input_length_sequence" length.
    """

    def __init__(self, root_dir, input_length_sequence, transform=None):
        super().__init__()
        self._input_length_sequence = input_length_sequence
        self._trajectories_data = load_npz_file(root_dir)
        # computing the length of each trajectory (# sequence) subtracting the input length sequence
        self._trajectories_length = [
            x.shape[0] - input_length_sequence for x, *_ in self._trajectories_data
        ]
        self._dataset_length = sum(self._trajectories_length)

        # Initialize the interval tree
        self._interval_tree = IntervalTree()
        current_end = 0
        for i, length in enumerate(self._trajectories_length):
            self._interval_tree[current_end : current_end + length] = i
            current_end += length

    def __len__(self):
        return self._dataset_length

    def __getitem__(self, idx):
        """
        Return a sequence of points in a trajectory at index idx with length _input_length_sequence
        """
        # Finding tuple with index idx belongs to which trajectory
        interval = list(self._interval_tree[idx])[0]
        trajectory_idx = interval.data
        positions, particle_type, *_material_property = self._trajectories_data[
            trajectory_idx
        ]

        trajectory_first_index = interval.begin
        tuple_idx_in_trajectory = idx - trajectory_first_index

        positions_in_sample = positions[
            tuple_idx_in_trajectory : tuple_idx_in_trajectory
            + self._input_length_sequence
        ]
        # positions_in_sample: (sequence_length, n_particle, dimension) -> (n_particle, sequence_length, dimension)
        positions_in_sample = np.transpose(positions_in_sample, (1, 0, 2))
        num_particle_in_sample = positions_in_sample.shape[0]
        label = positions[tuple_idx_in_trajectory + self._input_length_sequence]

        if _material_property:
            # We have material property in our features
            material_property = _material_property[0]
            assert len(material_property.shape) == 2
            assert material_property.shape[0] == num_particle_in_sample

            training_data = (
                (
                    torch.tensor(positions_in_sample).to(torch.float32).contiguous(),
                    torch.tensor(particle_type).contiguous(),
                    torch.tensor(material_property).to(torch.float32).contiguous(),
                    num_particle_in_sample,
                ),
                label
            )
        else:
            training_data = (
                (
                    torch.tensor(positions).to(torch.float32).contiguous(),
                    torch.tensor(particle_type).contiguous(),
                    num_particle_in_sample,
                ),
                label
            )
        
        return training_data
    
    @property
    def is_property_feature(self):
        return len(self._trajectories_data[0]) >= 3


def collate_fn(data):
    """Collate function for SamplesDataset.

    Args:
        data (list): List of tuples of the form ((positions, particle_type, n_particles_per_example), label).

    Returns:
        tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).
    """
    material_property_as_feature = True if len(data[0][0]) >= 4 else False
    position_list = []
    particle_type_list = []
    if material_property_as_feature:
        material_property_list = []
    n_particles_per_example_list = []
    label_list = []

    if material_property_as_feature:
        for (
            positions,
            particle_type,
            material_property,
            n_particles_per_example,
        ), label in data:
            position_list.append(positions)
            particle_type_list.append(particle_type)
            material_property_list.append(material_property)
            n_particles_per_example_list.append(n_particles_per_example)
            label_list.append(label)
    else:
        for (positions, particle_type, n_particles_per_example), label in data:
            position_list.append(positions)
            particle_type_list.append(particle_type)
            n_particles_per_example_list.append(n_particles_per_example)
            label_list.append(label)

    if material_property_as_feature:
        collated_data = (
            (
                torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
                torch.tensor(np.concatenate(particle_type_list)).contiguous(),
                torch.tensor(np.concatenate(material_property_list))
                .to(torch.float32)
                .contiguous(),
                torch.tensor(n_particles_per_example_list).contiguous(),
            ),
            torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous(),
        )
    else:
        collated_data = (
            (
                torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
                torch.tensor(np.concatenate(particle_type_list)).contiguous(),
                torch.tensor(n_particles_per_example_list).contiguous(),
            ),
            torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous(),
        )

    return collated_data


# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
# Dataset class for pytorch (mapping style)
class TrajectoriesDataset(Dataset):
    """
    Trajectories dataset where each trajectory is a series of tuples
        (positions, particle_type, material_property[optional]).
    positions' shape: (sequence_length, n_particle, dimension)
        where dimension is the number of coordinates (e.g. 2 for x and y)
    particle type shape: (n_particles)
    [optional] material_property: (n_particles, feature vector)
    """

    def __init__(self, root_dir, transform=None):
        self._trajectories_data = load_npz_file(root_dir)

    def __len__(self):
        """
        Return the length of the datest (i.e. # of trajectories)
        """
        return len(self._trajectories_data)

    def __getitem__(self, idx):
        """
        Return the full trajectory of index idx from the Trajectories Dataset
        Arguments:
            idx (int): index of the data (trajectory) within the dataset
        Returns:
            tuple[torch.Tensor]: (positions, particle_type, material_property (optional), n_particles_in_trajectory)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        positions, particle_type, *_material_property = self._trajectories_data[idx]
        # position: (sequence_length, n_particle, dimension) -> (n_particle, sequence_length, dimension)
        positions = np.transpose(positions, (1, 0, 2))
        num_particle_in_trajectory = positions.shape[0]
        # particle_type: (n_particles, ) each particle has a type
        assert particle_type.shape == (num_particle_in_trajectory,)

        if _material_property:
            # We have material property in our features
            material_property = _material_property[0]
            assert len(material_property.shape) == 2
            assert material_property.shape[0] == num_particle_in_trajectory

            return (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                torch.tensor(material_property).to(torch.float32).contiguous(),
                num_particle_in_trajectory,
            )
        else:
            return (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                num_particle_in_trajectory,
            )

    @property
    def is_property_feature(self):
        return len(self._trajectories_data[0]) >= 3


def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = SamplesDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def get_data_loader_by_trajectories(path):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = TrajectoriesDataset(path)
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, pin_memory=True
    )
