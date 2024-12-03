# Aref Hashemi & Aliakbar Izadkhah (2024)
# This code simulates a two-dimensional mixture of particles in a box

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['text.usetex'] = True


### parameters: length, time, and force values are scaled by particle radius (a), (a/g)^0.5, and mg, respectively.

# LJ interaction parameters
p = 4;
U0 = 0.05;
rm = 1.25;
R = 4;#2*np.power(2,1/p);

a = 1 ; # radius of particles

alpha, gamma = 0.14, 0.8
L = 10; # size of the box

### Initial conditions


# places N particles in a rectangular box of (lo) -- (hi)
def generate_equispaced_positions(N, lo, hi):
    L = hi[0]-lo[0]
    H = hi[1]-lo[1]
    # Determine the number of points along each axis
    n_cols = int(np.ceil(np.sqrt(N*(L/H))))
    n_rows = int(np.ceil(N/n_cols))

    # Generate equispaced points along each axis
    x = np.linspace(lo[0], hi[0], n_cols, endpoint=False)
    y = np.linspace(lo[1], hi[1], n_rows, endpoint=False)
    
    # Create a grid of points
    xv, yv = np.meshgrid(x, y)
    grid_points = np.vstack([xv.ravel(), yv.ravel()]).T

    # Select the first N points
    positions = grid_points[:N]

    return positions

### Total pair interaction force on particle i
dUdrm = -4*p*U0/rm*(2*np.power(2/rm,2*p)-np.power(2/rm,p)); # dU(rm)/dr
def Fi(x,i):
    N = len(x);
    diff = x[i,:]-x # vector of difference between all points and point i
    r = np.sqrt(np.power(diff[:,0],2)+np.power(diff[:,1],2)).reshape((N,1)); # vector of distance between all points and point i
    
    # remove the ith rows from diff and r vectors (to avoid division by zero)
    rMod = np.delete(r,i,0);
    diffMod = np.delete(diff,i,0);

    # logical conditions to find the vector of forces
    ForceVec = -dUdrm*diffMod/rMod*(rMod<=rm)+4*p*U0*(2*np.power(2/rMod,2*p)-np.power(2/rMod,p))*diffMod/np.power(rMod,2)*np.logical_and(rm<rMod, rMod<=R);
    return np.sum(ForceVec,axis=0);

def FVec(x):
    N = len(x);
    DIM = (N,2);
    FVec = np.zeros(DIM);
    for i in range(N):
        FVec[i,:] = Fi(x,i);

    return FVec;

ex = np.array([1,0])
ey = np.array([0,1])
def collision(wallpos, direction, nhat, dt, x, v, index, gamma, alpha, F_ext):
    dt_c = (wallpos-x[index,direction])/v[index,direction]
    dt_r = dt-dt_c
    xc  = x[index,:]+dt_c*v[index,:]
    vbc = v[index,:]+dt_c*(-9/2*alpha*v[index,:]+(F_ext[index,:]+Fi(x,index)))
    vac = np.sqrt(gamma)*(vbc-2*(np.dot(vbc,nhat))*nhat)
    x[index,:] = xc
    vNew = vac+dt_r*(-9/2*alpha*vac+(F_ext[index,:]+Fi(x,index)))
    xNew = xc+dt_r*vac

    return (vNew,xNew)


### Simulation

num_trajectories = 2
data_category = 'test'
np.random.seed(12357) # train: 1; valid: 123; test: 12357
for tr in range(num_trajectories):
    N = np.random.randint(27, 30, 1).item()
    DIM = (N,2);
    ### Extenral force
    F_ext = np.zeros(DIM);

    lo = [np.random.uniform(-L+a,-a,1)[0],np.random.uniform(-0.75*L,-0.5*L,1)[0]]
    box_sizex = np.random.uniform(0.75*L,L,1)[0]
    box_sizey = 4*N/(box_sizex)
    if lo[1]+box_sizey > L-a:
        lo[1] -= lo[1]+box_sizey-(L-a)
    hi = np.add(lo,np.array([box_sizex,box_sizey]))
    x = generate_equispaced_positions(N,lo,hi)
    vp0x = 0.5*np.random.uniform(-np.power(2*L,0.5),np.power(2*L,0.5))
    vp0y = 0
    v = 2*np.tile([vp0x,vp0y],(N,1))

    # print(f'v[:10,:]=\n{v[:10,:]}')
    Lx = 5;# in
    Ly = 5;# in
    plt.figure(figsize=[Lx, Ly])
    ax = plt.axes([0.15, 0.15, 0.8, 0.8], xlim=(-1, 1), ylim=(-1, 1))
    points_whole_ax = Lx*0.8*72# 1 point = dpi / 72 pixels
    points_radius = 1/L*points_whole_ax

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xticks([-1,-0.5,0,0.5,1])
    plt.yticks([-1,-0.5,0,0.5,1])
    curve = ax.scatter(10, 10, s=points_radius**2, color='r')

    dt = 0.01; # time step
    tf = 15  ; # total simulation time
    tM = int(tf/dt); # number of time steps
    t = np.linspace(0,tf,tM);
    position = np.zeros((tM,DIM[0],DIM[1]))
    position[0,:,:] = x
    radius = a/L
    particle_properties = np.full((N,), radius)

    for n in range(tM-1):
        # Time marching

        ### External forces
        ## constant force
        F_ext[:,1] = -1; # constant force (scaled by mg) along y direction
        ## Fixed point at the center
        # F_ext = -0.01*x; # stable fixed point at (0,0)

        ## potential field phi=sin(m*pi*x/L)*exp(-k*(y/L)^2)
        # m = 1;k = 5;F = 10;
        # F_ext[:,0] = -F*m*np.pi/L*np.cos(m*np.pi*x[:,0]/L)*np.exp(-k*np.power(x[:,1]/L,2));
        # F_ext[:,1] = -F*np.sin(m*np.pi*x[:,0]/L)*(-2*k*x[:,1]/np.power(L,2))*np.exp(-k*np.power(x[:,1]/L,2));
    
        # ## potential field phi = exp((x/L+a)^2+(y/L+b)^2)
        # a = 1/4; b = 0;F = 10;
        # F_ext[:,0] = -F*2*(x[:,0]/L+a)/L*np.exp(np.power(x[:,0]/L+a,2)+np.power(x[:,1]/L+b,2));
        # F_ext[:,1] = -F*2*(x[:,1]/L+b)/L*np.exp(np.power(x[:,0]/L+a,2)+np.power(x[:,1]/L+b,2));
    
        vNew = v+dt*(-9/2*alpha*v+(F_ext+FVec(x)))
        xNew = x+dt*v

        # Boundary conditions
        indices_hix, indices_lox = np.where(xNew[:,0]>L-a), np.where(xNew[:,0]<-L+a)
        indices_hiy, indices_loy = np.where(xNew[:,1]>L-a), np.where(xNew[:,1]<-L+a)
        # Right wall
        if indices_hix[0].size>0:
            for index in indices_hix[0]:
                (vNew[index,:],xNew[index,:]) = collision(L-a, 0, -ex, dt, x, v, index, gamma, alpha, F_ext)
        # Left wall
        if indices_lox[0].size>0:
            for index in indices_lox[0]:
                (vNew[index,:],xNew[index,:]) = collision(-L+a, 0, ex, dt, x, v, index, gamma, alpha, F_ext)
        # Top wall
        if indices_hiy[0].size>0:
            for index in indices_hiy[0]:
                (vNew[index,:],xNew[index,:]) = collision(L-a, 1, -ey, dt, x, v, index, gamma, alpha, F_ext)
        # Bottom wall
        if indices_loy[0].size>0:
            for index in indices_loy[0]:
                (vNew[index,:],xNew[index,:]) = collision(-L+a, 1, ey, dt, x, v, index, gamma, alpha, F_ext)

        v, x = vNew, xNew
        position[n+1,:,:] = x
        # print(format(t[n], '.2f'))
        curve.remove()
        t_pause = 0.001;
        print(f'time={(n+1)*dt:.2f}')
        curve = ax.scatter(x[:,0]/L,x[:,1]/L, s=points_radius**2, color='r', edgecolor='k')
        # for i in range(len(x)):
        #     curve = ax.text(x[i,0],x[i,1], str(i))
        
        plt.pause(t_pause)


    plt.show(block=False)
    plt.close()
    
    data_tuple = np.empty(2, dtype=object)
    data_tuple[0] = position[0:-1:2,:,:]/L # normalized
    data_tuple[1] = particle_properties
    np.save(f"{data_category}{tr}.npy", data_tuple)


data_dict = {}
for tr in range(num_trajectories):
    # Load the data from each .npy file
    data = np.load(f"{data_category}{tr}.npy", allow_pickle=True)
    
    # Assign the loaded data to the dictionary with the desired naming convention
    data_dict[f"simulation_trajectory_{tr}"] = data

# Save all data into a single .npz file
import os
np.savez(f"{data_category}.npz", **data_dict)
os.system("rm *.npy")

if data_category == 'train':
    # train_data = np.load('train.npz', allow_pickle=True)
    # num_trajectories = len(train_data.files)
    # vel_mean_vec = np.zeros((num_trajectories,2))
    # vel_std_vec = np.zeros((num_trajectories,2))
    # acc_mean_vec = np.zeros((num_trajectories,2))
    # acc_std_vec = np.zeros((num_trajectories,2))
    # for idx in range(len(train_data.files)):
    #     st=train_data[f'simulation_trajectory_{idx}']
    #     position = st[0]
    #     num_steps = position.shape[0]
    #     N = position.shape[1]
    #     vel = np.zeros((num_steps,N,2))
    #     acc = np.zeros((num_steps,N,2))
    #     for i in range(1,num_steps):
    #         vel[i,:,:] = (position[i,:,:]-position[i-1,:,:])/dt

    #     for i in range(1,num_steps-1):
    #         acc[i,:,:] = (position[i+1,:,:]-2*position[i,:,:]+position[i-1,:,:])/(dt**2)
        
    #     vel = vel[2:,:,:]
    #     acc = acc[1:-1,:,:]
    #     vel_mean_vec[idx], vel_std_vec[idx] = np.mean(vel, axis=(0,1)), np.std(vel, axis=(0,1))
    #     acc_mean_vec[idx], acc_std_vec[idx] = np.mean(acc, axis=(0,1)), np.std(acc, axis=(0,1))

    # vel_mean = np.mean(vel_mean_vec, axis=0)
    # acc_mean = np.mean(acc_mean_vec, axis=0)
    # vel_std = np.mean(vel_std_vec**2+(vel_mean_vec-vel_mean)**2, axis=0)
    # acc_std = np.mean(acc_std_vec**2+(acc_mean_vec-acc_mean)**2, axis=0)

    import json
    data = np.load('train.npz', allow_pickle=True)
    radii = []
    for example in range(len(data.files)):
        radii.append(data[f'simulation_trajectory_{example}'][1])

    max_radius = np.max(np.concatenate(radii))
    default_connectivity_radius = 4*max_radius
    # Define the metadata dictionary
    metadata = {
        "bounds": [[-1, 1], [-1, 1]],
        "default_connectivity_radius": default_connectivity_radius,
        "boxSize": L
    }

    # Write the metadata dictionary to a JSON file
    with open('metadata.json', 'w') as json_file:
        json.dump(metadata, json_file)
