import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def load_npz_data(file_path):
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
        self._trajectories_data = load_npz_data(root_dir)

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
