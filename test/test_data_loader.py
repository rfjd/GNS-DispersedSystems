import unittest
import numpy as np
from gns.data_loader import load_npz_data, TrajectoriesDataset


class DataLoaderTest(unittest.TestCase):
    def test_valid_file(self):
        # Create a temporary .npz file with sample data
        sample_data = {
            "traj1": np.array([[1, 2], [3, 4]]),
            "traj2": np.array([[5, 6], [7, 8]]),
        }
        np.savez("temp.npz", **sample_data)

        # Test loading the data
        loaded_data = load_npz_data("temp.npz")

        # Check if the loaded data matches the sample data
        self.assertEqual(len(loaded_data), 2)
        self.assertTrue(np.array_equal(loaded_data[0], sample_data["traj1"]))
        self.assertTrue(np.array_equal(loaded_data[1], sample_data["traj2"]))

    def test_invalid_file(self):
        # Test loading a non-existent file
        with self.assertRaises(FileNotFoundError):
            load_npz_data("nonexistent.npz")

    def test_trajectories_dataset_without_material_property(self):
        """
        Dataset with 30 particle, no material property, and 100,000 sequence with 3 dimension x,y,theta
        """
        dataset = TrajectoriesDataset(root_dir="data/train.npz")
        assert dataset._trajectories_data is not None
        assert not dataset.is_property_feature

        # check the shapes
        data = dataset[0]
        assert len(data) == 3  # position, material_type, n_particle
        assert data[-1] == 30
        assert data[0].shape == (30, 100000, 3)
        assert data[1].shape == (30,)

    def test_trajectories_dataset_with_material_property(self):
        """
        Dataset with 30 particle, with material property, and 1,000 sequence with 3 dimension x,y,theta
        Material property is a vector with 4 features
        """
        dataset = TrajectoriesDataset(root_dir="data/temp_material_property.npz")
        assert dataset._trajectories_data is not None
        assert dataset.is_property_feature

        # check the shapes
        data = dataset[0]
        assert len(data) == 4  # position, material_type, material_property, n_particle
        assert data[-1] == 30
        assert data[0].shape == (30, 1000, 3)
        assert data[1].shape == (30,)
        assert data[2].shape == (30, 4)
