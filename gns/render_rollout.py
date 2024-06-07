# Aref Hashemi & Aliakbar Izadkhah 2024
import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from pyevtk.hl import pointsToVTK

flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout `.pkl` file")
flags.DEFINE_integer("step_stride", 10, help="Stride of steps to skip")

FLAGS = flags.FLAGS

TYPE_TO_COLOR = {
    0: "red",
    1: "black",  # Boundary particles.
    2: "green",
    3: "magenta",
    4: "gold",
    5: "blue",
}


class Render():
    """
    Render rollout data into gif or vtk files
    """

    def __init__(self, input_dir, input_name):
        """
            Initialize render class

        Args:
            input_dir (str): Directory where rollout.pkl are located
            input_name (str): Name of rollout `.pkl` file
        """
        # Texts to describe rollout cases for data and render
        rollout_cases = [["ground_truth_rollout", "Reality"], ["predicted_rollout", "GNS"]]
        self.rollout_cases = rollout_cases
        self.input_dir = input_dir
        self.input_name = input_name
        self.output_dir = input_dir
        self.output_name = input_name

        # Get trajectory
        with open(f"{self.input_dir}{self.input_name}.pkl", "rb") as file:
            rollout_data = pickle.load(file)
        self.rollout_data = rollout_data
        trajectory = {}
        for rollout_case in rollout_cases:
            trajectory[rollout_case[0]] = np.concatenate(
                [rollout_data["initial_positions"], rollout_data[rollout_case[0]]], axis=0
            )
        self.trajectory = trajectory
        self.loss = self.rollout_data['loss'].item()

        # Trajectory information
        self.dims = trajectory[rollout_cases[0][0]].shape[2]-1
        self.num_particles = trajectory[rollout_cases[0][0]].shape[1]
        self.num_steps = trajectory[rollout_cases[0][0]].shape[0]
        self.boundaries = rollout_data["metadata"]["bounds"]
        self.particle_type = rollout_data["particle_types"]

    def color_map(self):
        """
        Get color map array for each particle type for visualization
        """
        # color mask for visualization for different material types
        color_map = np.empty(self.num_particles, dtype="object")
        for material_id, color in TYPE_TO_COLOR.items():
            print(material_id, color)
            color_index = np.where(np.array(self.particle_type) == material_id)
            print(color_index)
            color_map[color_index] = color
        color_map = list(color_map)
        return color_map

    def color_mask(self):
        """
        Get color mask and corresponding colors for visualization
        """
        color_mask = []
        for material_id, color in TYPE_TO_COLOR.items():
            mask = np.array(self.particle_type) == material_id
            if mask.any() == True:
                color_mask.append([mask, color])
        return color_mask

    def render_gif_animation(self, timestep_stride=3, vertical_camera_angle=20, viewpoint_rotation=0.5):
        """
        Render `.gif` animation from `.pkl` trajectory data.

        Args:
            timestep_stride (int): Stride of steps to skip.
            vertical_camera_angle (float): Vertical camera angle in degree
            viewpoint_rotation (float): Viewpoint rotation in degree

        Returns:
            gif format animation
        """
        # Get boundary of simulation
        xboundary = self.boundaries[0]
        yboundary = self.boundaries[1]
        
        # Init figure
        # figsize, margins and spacing in inches
        subplot_width = subplot_height = 3
        left_margin = 0.75
        right_margin = 0.5
        top_margin = 0.5
        bottom_margin = 0.5
        horizontal_spacing = 0.75

        # calculate point size for scatter plot
        boxSize = self.rollout_data["metadata"]["boxSize"]
        points_whole_ax = subplot_width*0.8*72# 1 point = dpi / 72 pixels
        points_radius = 2/boxSize*points_whole_ax
        point_size = points_radius**2
        
        # Calculate the overall figure size
        fig_width = left_margin + 2*subplot_width + horizontal_spacing + right_margin
        fig_height = bottom_margin + subplot_height + top_margin

        # Create the figure
        fig = plt.figure(figsize=(fig_width, fig_height))
        axes = []
        for j in range(2):
            ax = fig.add_axes([(left_margin+j*(subplot_width+horizontal_spacing))/fig_width, bottom_margin/fig_height, subplot_width/fig_width, subplot_height/fig_height])
            ax.set_xlim([float(xboundary[0]), float(xboundary[1])])
            ax.set_ylim([float(yboundary[0]), float(yboundary[1])])
            axes.append(ax)
        
        # Define datacase name
        trajectory_datacases = [self.rollout_cases[0][0], self.rollout_cases[1][0]]
        render_datacases = [self.rollout_cases[0][1], self.rollout_cases[1][1]]
        
        # Get color mask for visualization
        color_mask = self.color_mask()

        # Fig creating function for 2d
        
        def animate(i):
            print(f"Render step {i}/{self.num_steps}")
            for j, datacase in enumerate(trajectory_datacases):
                axes[j].cla()
                for mask, color in color_mask:
                    axes[j].scatter(self.trajectory[datacase][i][mask, 0],
                                    self.trajectory[datacase][i][mask, 1], s=point_size, color=color)
                axes[j].grid(True, which='both')
                axes[j].set_title(render_datacases[j])
                axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
            # fig.suptitle(f"{i}/{self.num_steps}, Total MSE: {self.loss:.2e}")

        # Creat animation
        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, self.num_steps, timestep_stride), interval=10)

        ani.save(f'{self.output_dir}{self.output_name}.gif', dpi=100, fps=30, writer='imagemagick')
        print(f"Animation saved to: {self.output_dir}{self.output_name}.gif")


def main(_):
    if not FLAGS.rollout_dir:
        raise ValueError("A `rollout_dir` must be passed.")
    if not FLAGS.rollout_name:
        raise ValueError("A `rollout_name`must be passed.")

    render = Render(input_dir=FLAGS.rollout_dir, input_name=FLAGS.rollout_name)
              
    render.render_gif_animation(
        timestep_stride=FLAGS.step_stride,
        vertical_camera_angle=20,
        viewpoint_rotation=0.3
    )

if __name__ == '__main__':
    app.run(main)

