import numpy as np
import json

np.random.seed(seed=0)

def generate_data(mode):
    data_dict = {}
    spatial_dimension = 2
    num_examples = 2
    num_particles_per_example = np.random.randint(5, 10, (num_examples,))
    num_time_steps = 100
    for example in range(num_examples):
        data_tuple = np.empty(2, dtype=object)
        num_particles = num_particles_per_example[example]
        position = np.random.rand(num_time_steps, num_particles, spatial_dimension)
        particle_type = np.full((num_particles,),0,dtype=np.int32)
        data_tuple[0] = position
        data_tuple[1] = particle_type

        data_dict[f"simulation_trajectory_{example}"] = data_tuple

    np.savez(f"{mode}.npz", **data_dict)

    if mode == "train":
        metadata = {
            "bounds": [[0, 1], [0, 1]],
            "sequence_length": num_time_steps,
            "default_connectivity_radius": 0.1,
            "dim": 2,
            "vel_mean": [0.01,0.05],
            "vel_std": [0.5,0.1],
            "acc_mean": [0.002,0.003],
            "acc_std": [0.03,0.015],
            "boxSize": 1
        }

        # Write the metadata dictionary to a JSON file
        with open('metadata.json', 'w') as json_file:
            json.dump(metadata, json_file)


generate_data("train")
generate_data("test")
generate_data("valid")

import os
os.system("mkdir -p sampleDATA")
os.system("mv metadata.json sampleDATA/")
os.system("mv train.npz sampleDATA/")
os.system("mv test.npz sampleDATA/")
os.system("mv valid.npz sampleDATA/")