# Aref Hashemi and Aliakbar Izadkhah 2024
# Simulation of two-dimensional rotating smooth hard spheres in a square box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['text.usetex'] = True
from absl import flags
from absl import app
import os
import json

flags.DEFINE_string("data_category", "temp", help="category of the simulation: train, valid, test")
flags.DEFINE_integer("num_trajectories", 2, help="number of trajectories")
flags.DEFINE_integer("seed_number", 1, help="seed number")
FLAGS = flags.FLAGS

# Constants
L = 30     # Size of the square box
N = 30     # Number of particles
radius = 1 # Radius of each particle
mass = 1   # Mass of each particle
g = 1      # Gravitational acceleration
dt = 0.01  # Time step
tf = 10    # Simulation time
tM = int(tf/dt)
restitution_particle = 0.8  # Coefficient of restitution for particle-particle collisions
restitution_wall = 0.5  # Coefficient of restitution for particle-wall collisions

class Sphere:
    def __init__(self, x, y, vx, vy, radius, mass, theta, omega=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.mass = mass
        self.omega = omega  # Angular velocity
        self.theta = theta  # Rotation angle

def handle_particle_collisions(sphere1, sphere2):
    dx = sphere1.x - sphere2.x
    dy = sphere1.y - sphere2.y
    distance = np.hypot(dx, dy)

    if distance < sphere1.radius + sphere2.radius:
        angle = np.arctan2(dy, dx)
        total_mass = sphere1.mass + sphere2.mass
        
        # Normal vector
        nx, ny = np.cos(angle), np.sin(angle)

        # Tangent vector
        tx, ty = -ny, nx

        # Velocity components in normal and tangential directions
        v1n = sphere1.vx*nx + sphere1.vy*ny
        v1t = sphere1.vx*tx + sphere1.vy*ty
        v2n = sphere2.vx*nx + sphere2.vy*ny
        v2t = sphere2.vx*tx + sphere2.vy*ty

        # New normal velocities after collision with restitution
        v1n_new = (v1n*(sphere1.mass-restitution_particle*sphere2.mass)+(1+restitution_particle)*sphere2.mass*v2n)/total_mass
        v2n_new = (v2n*(sphere2.mass-restitution_particle*sphere1.mass)+(1+restitution_particle)*sphere1.mass*v1n)/total_mass

        # Convert scalar normal and tangential velocities into vectors
        sphere1.vx = v1n_new*nx + v1t*tx
        sphere1.vy = v1n_new*ny + v1t*ty
        sphere2.vx = v2n_new*nx + v2t*tx
        sphere2.vy = v2n_new*ny + v2t*ty

        # Angular velocity changes due to tangential velocity exchange
        delta_vt1 = v1t-v2t
        delta_vt2 = v2t-v1t
        sphere1.omega += delta_vt1/sphere1.radius
        sphere2.omega += delta_vt2/sphere2.radius

        # Adjust positions to avoid overlap
        overlap = 0.5*(sphere1.radius+sphere2.radius-distance)
        sphere1.x += np.cos(angle)*overlap/2
        sphere1.y += np.sin(angle)*overlap/2
        sphere2.x -= np.cos(angle)*overlap/2
        sphere2.y -= np.sin(angle)*overlap/2

def handle_wall_collisions(sphere, L):
    if sphere.x-sphere.radius < 0:
        sphere.x = sphere.radius
        sphere.vx = -sphere.vx*restitution_wall
        sphere.omega = -sphere.omega
    if sphere.x+sphere.radius > L:
        sphere.x = L-sphere.radius
        sphere.vx = -sphere.vx*restitution_wall
        sphere.omega = -sphere.omega
    if sphere.y-sphere.radius < 0:
        sphere.y = sphere.radius
        sphere.vy = -sphere.vy*restitution_wall
        sphere.omega = -0.5*sphere.omega # Trying to 'calm' the particles down!
    if sphere.y+sphere.radius > L:
        sphere.y = L-sphere.radius
        sphere.vy = -sphere.vy*restitution_wall
        sphere.omega = -sphere.omega

def writePos(particles, position, time_step):
    for i, p in enumerate(particles):
        position[time_step,i,:] = np.array([p.x,p.y,p.theta])


def animate(frame, particles, circles, dots, position):
    # print(f'frame={frame}')
    for i, p in enumerate(particles):
        # Update positions
        p.vy -= g*dt  # Gravity
        p.x += p.vx*dt
        p.y += p.vy*dt
        p.theta += p.omega*dt # Update rotation angle
        
        # Handle wall collisions
        handle_wall_collisions(p, L)

        # Handle particle collisions
        for j in range(i+1, N):
            handle_particle_collisions(p, particles[j])

        writePos(particles, position, frame)
        # Update circle positions and rotation dots
        circles[i].center = (p.x, p.y)
        dots[i].set_data(p.x + p.radius * np.cos(p.theta), p.y + p.radius * np.sin(p.theta))
        
    return circles+dots

####### Simulation
def main(_):
    global N, radius, mass, g, dt, tM, L, restitution_particle, restitution_wall
    data_category = FLAGS.data_category
    num_trajectories = FLAGS.num_trajectories
    np.random.seed(FLAGS.seed_number)
    for tr in range(num_trajectories):
        # Initialize particles
        particles = []
        for _ in range(N):
            x = np.random.uniform(radius, L-radius)
            y = np.random.uniform(radius, L-radius)
            vx = np.random.uniform(-1, 1)
            vy = np.random.uniform(-1, 1)
            theta = np.random.uniform(0, 2*np.pi)
            particles.append(Sphere(x, y, vx, vy, radius, mass, theta))

        position = np.zeros((tM,N,3)) # x, y, theta
        particle_type = np.full((N,),5,dtype=np.int32)

        # Create figure
        fig, ax = plt.subplots()
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        circles = [plt.Circle((p.x, p.y), p.radius, ec='b', fill=False) for p in particles]
        dots = [ax.plot(p.x+p.radius*np.cos(p.theta), p.y+p.radius*np.sin(p.theta), 'ro')[0] for p in particles]
        for circle in circles:
            ax.add_patch(circle)

        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,tM), interval=1, repeat=False, fargs=(particles, circles, dots, position))
        plt.show(block=False)
        plt.pause(1)
        ani.event_source.stop()
        del ani
        plt.close()
    

        data_tuple = np.empty(2, dtype=object)
        data_tuple[0] = position/L # normalized
        data_tuple[1] = particle_type
        np.save(f"{data_category}{tr}.npy", data_tuple)


    data_dict = {}
    for tr in range(num_trajectories):
        # Load the data from each .npy file
        data = np.load(f"{data_category}{tr}.npy", allow_pickle=True)
    
        # Assign the loaded data to the dictionary with the desired naming convention
        data_dict[f"simulation_trajectory_{tr}"] = data

    # Save all data into a single .npz file
    np.savez(f"{data_category}.npz", **data_dict)
    os.system("rm *.npy")


    if data_category == 'train':
        train_data = np.load('train.npz', allow_pickle=True)
        num_trajectories = len(train_data.files)
        vel_mean_vec = np.zeros((num_trajectories,3))
        vel_std_vec = np.zeros((num_trajectories,3))
        acc_mean_vec = np.zeros((num_trajectories,3))
        acc_std_vec = np.zeros((num_trajectories,3))
    
        for idx in range(len(train_data.files)):
            st=train_data[f'simulation_trajectory_{idx}']
            position = st[0]
            num_steps = position.shape[0]
            N = position.shape[1]
            vel = acc = np.zeros((num_steps,N,3))
            for i in range(1,num_steps):
                vel[i,:,:] = (position[i,:,:]-position[i-1,:,:])/dt

            for i in range(1,num_steps-1):
                acc[i,:,:] = (position[i+1,:,:]-2*position[i,:,:]+position[i-1,:,:])/(dt**2)
        
            vel = vel[2:,:,:]
            acc = acc[1:-1,:,:]
        
            vel_mean_vec[idx], vel_std_vec[idx] = np.mean(vel, axis=(0,1)), np.std(vel, axis=(0,1))
            acc_mean_vec[idx], acc_std_vec[idx] = np.mean(acc, axis=(0,1)), np.std(acc, axis=(0,1))

        vel_mean, acc_mean = np.mean(vel_mean_vec, axis=0), np.mean(acc_mean_vec, axis=0) 
        vel_std, acc_std = np.mean(vel_std_vec**2+(vel_mean_vec-vel_mean)**2, axis=0), np.mean(acc_std_vec**2+(acc_mean_vec-acc_mean)**2, axis=0)

        # Define the metadata dictionary
        metadata = {
            "bounds": [[0, L], [0, L]],
            "sequence_length": num_steps,
            "default_connectivity_radius": 4,
            "dim": 2,
            "dt": dt,
            "vel_mean": vel_mean.tolist(),
            "vel_std": vel_std.tolist(),
            "acc_mean": acc_mean.tolist(),
            "acc_std": acc_std.tolist()
        }

        # Write the metadata dictionary to a JSON file
        with open('metadata.json', 'w') as json_file:
            json.dump(metadata, json_file)


if __name__ == "__main__":
    app.run(main)
