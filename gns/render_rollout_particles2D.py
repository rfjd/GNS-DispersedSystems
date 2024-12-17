# Aref Hashemi & Aliakbar Izadkhah 2024
import pickle
from absl import app
from absl import flags
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.io import savemat

flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout .pkl file")
flags.DEFINE_integer("step_stride", 10, help="Stride of steps to skip")

FLAGS = flags.FLAGS

class Render():
    """
    Render rollout data into gif or vtk files
    """

    def __init__(self, input_dir, input_name):
        rollout_cases = ["ground_truth_rollout", "predicted_rollout"]
        self.rollout_cases = rollout_cases
        self.titles = ["Ground Truth", "Graph Neural Network"]
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
            trajectory[rollout_case] = np.concatenate([rollout_data["initial_positions"], rollout_data[rollout_case]], axis=0)
        self.trajectory = trajectory

        # Trajectory information
        self.num_steps = trajectory[rollout_cases[0]].shape[0]
        self.boundaries = rollout_data["metadata"]["bounds"]
        self.particle_properties = rollout_data["particle_properties"]

    def render_gif_animation(self, timestep_stride=3, particle_radii_min=0.02, particle_radii_max=0.05):
        # Get boundary of simulation
        xboundary = self.boundaries[0]
        yboundary = self.boundaries[1]
        
        # Init figure
        # figsize, margins and spacing in inches
        subplot_width = subplot_height = 3
        left_margin = 0.15
        right_margin = 0.15
        top_margin = 0.3
        bottom_margin = 0.15
        horizontal_spacing = 0.25

        if len(list(self.particle_properties.shape)) == 1:
            particle_radii = self.particle_properties # shape = (num_particles,)
        else:
            particle_radii = self.particle_properties[:,0] # shape = (num_particles,)
            
        # calculate point size for scatter plot
        boxSize = self.rollout_data["metadata"]["boxSize"]
        points_whole_ax = subplot_width*0.8*72# 1 point = dpi / 72 pixels
        points_radius = particle_radii*points_whole_ax
        point_size = points_radius**2

        norm = mpl.colors.Normalize(vmin=particle_radii_min, vmax=particle_radii_max)
        colormap = plt.cm.jet
        
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
            ax.set_facecolor([0.9, 0.95, 1])  # Set liquid-like background color
            axes.append(ax)
        
        # Define datacase name
        trajectory_datacases = [self.rollout_cases[0], self.rollout_cases[1]]
        
        def animate(i):
            print(f"Render step {i}/{self.num_steps}")
            for j, datacase in enumerate(trajectory_datacases):
                axes[j].cla()
                # Map particle radii to colormap
                colors = colormap(norm(particle_radii))  # Apply colormap based on radii
                axes[j].scatter(self.trajectory[datacase][i,:,0], self.trajectory[datacase][i,:,1], s=point_size, c=colors, edgecolor="k")
                axes[j].grid(True, which='both')
                axes[j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                axes[j].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                axes[j].set_title(self.titles[j])
                axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])

        # Creat animation
        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, self.num_steps, timestep_stride), interval=5)

        ani.save(f'{self.output_dir}{self.output_name}.gif', dpi=100, fps=30, writer='imagemagick')
        print(f"Animation saved to: {self.output_dir}{self.output_name}.gif")


    def save_trajectory_to_mat(self):
        trajectory_data = {
            "ground_truth_rollout": self.trajectory["ground_truth_rollout"],
            "predicted_rollout": self.trajectory["predicted_rollout"],
            "particle_radii": self.particle_properties,
            "metadata": self.rollout_data["metadata"]
        }
        savemat(f'{self.output_dir}{self.output_name}.mat', trajectory_data)
        print(f"Trajectory saved to: {self.output_dir}{self.output_name}.mat")


def main(_):
    if not FLAGS.rollout_dir:
        raise ValueError("A rollout_dir must be passed.")
    if not FLAGS.rollout_name:
        raise ValueError("A rollout_namemust be passed.")

    render = Render(input_dir=FLAGS.rollout_dir, input_name=FLAGS.rollout_name)          
    render.render_gif_animation(timestep_stride=FLAGS.step_stride)
    render.save_trajectory_to_mat()
    
if __name__ == '__main__':
    app.run(main)
