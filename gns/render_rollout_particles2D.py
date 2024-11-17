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
        self.dims = trajectory[rollout_cases[0][0]].shape[2]
        self.num_particles = trajectory[rollout_cases[0][0]].shape[1]
        self.num_steps = trajectory[rollout_cases[0][0]].shape[0]
        self.boundaries = rollout_data["metadata"]["bounds"]
        self.particle_properties = rollout_data["particle_properties"]

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
        # print(self.particle_properties)
        if len(list(self.particle_properties.shape)) == 1:
            particle_radii = self.particle_properties # shape = (num_particles,)
        else:
            particle_radii = self.particle_properties[:,0] # shape = (num_particles,)

        # calculate point size for scatter plot
        boxSize = self.rollout_data["metadata"]["boxSize"]
        points_whole_ax = subplot_width*0.8*72# 1 point = dpi / 72 pixels
        points_radius = particle_radii*points_whole_ax
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
        # print(f"trajectory_datacases: {trajectory_datacases}") # grund truth rollout, predicted rollout
        # print(f"self.trajectory.shape={self.trajectory[trajectory_datacases[0]].shape}")
        # Get color mask for visualization
        color_mask = [] #self.color_mask()

        # Fig creating function for 2d
        
        def animate(i):
            print(f"Render step {i}/{self.num_steps}")
            for j, datacase in enumerate(trajectory_datacases):
                axes[j].cla()
                # for mask, color in color_mask:
                axes[j].scatter(self.trajectory[datacase][i,:,0],
                                    self.trajectory[datacase][i,:,1], s=point_size, color="blue", edgecolor="k")
                axes[j].grid(True, which='both')
                axes[j].set_title(render_datacases[j])
                axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
            # fig.suptitle(f"{i}/{self.num_steps}, Total MSE: {self.loss:.2e}")

        # Creat animation
        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, self.num_steps, timestep_stride), interval=5)

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

