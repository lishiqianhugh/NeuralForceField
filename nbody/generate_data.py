import rebound
import numpy as np
from scipy.stats.qmc import LatinHypercube
from utils.util import vis_nbody_traj

np.random.seed(1)

def n_body_simulation(n, steps=3000, dt=0.02, save_fps=300, sample=None, speed_res=0, max_body=3, comet=False):
    """
    General n-body simulator
    Args:
        n (int): number of planets / comets
        steps (int): simulation steps
        dt (float): time step
        max_body (int): maximum number of samll bodies in the simulation
    """
    
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.dt = dt
    masses = [5.0]
    xs = [0]
    ys = [0]
    zs = [0]
    vxs = [0]
    vys = [0]
    vzs = [0]
    for i in range(len(masses)):
        sim.add(m=masses[i], x=xs[i], y=ys[i], z=zs[i], vx=vxs[i], vy=vys[i], vz=vzs[i])

    for i in range(n):
        angle_xy = sample[i, 0]  
        angle_z = sample[i, 1] 
        radius = sample[i, 2] 
        
        x = radius * np.cos(angle_z) * np.cos(angle_xy)
        y = radius * np.cos(angle_z) * np.sin(angle_xy)
        z = radius * np.sin(angle_z)
        if comet:
            speed = np.sqrt(2*sim.particles[0].m / radius)
        else:
            speed = np.sqrt(sim.particles[0].m / radius)
        
        vx = -speed * np.cos(angle_z) * np.sin(angle_xy)
        vy = speed * np.cos(angle_z) * np.cos(angle_xy)
        vz = 0 

        mass = sample[i, 3]
        masses.append(mass)
        sim.add(m=mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)


    traj_masses = [[] for _ in range(n + 1)]
    positions = [[] for _ in range(n + 1)]
    velocities = [[] for _ in range(n + 1)]
    accs = [[] for _ in range(n + 1)]
    for t in range(steps):
        # detect collision
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                dx = sim.particles[i].x - sim.particles[j].x
                dy = sim.particles[i].y - sim.particles[j].y
                dz = sim.particles[i].z - sim.particles[j].z
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                if distance < 1e-1:
                    return None

        for i in range(n + 1):
            traj_masses[i].append([sim.particles[i].m])
            positions[i].append([sim.particles[i].x, sim.particles[i].y, sim.particles[i].z])
            velocities[i].append([sim.particles[i].vx, sim.particles[i].vy, sim.particles[i].vz])
            accs[i].append([sim.particles[i].ax, sim.particles[i].ay, sim.particles[i].az])

        sim.integrate(sim.t + dt)

    data = np.concatenate([np.array(traj_masses), np.array(positions), np.array(velocities), np.array(accs)], axis=-1) # [body_num, steps, 10]
    # downsampling steps to 100
    data = data[:, ::save_fps, :]
    data = data.transpose(1, 0, 2)  # [steps, body_num, 10]
    # padding
    if n < max_body:
        padding_shape = (data.shape[0], max_body - n, data.shape[2])
        padding = np.zeros(padding_shape)
        data = np.concatenate([data, padding], axis=1)
    # vis_nbody_traj(data)

    return data

def generate_train_data():
    # generate training sequences
    all_data = []
    seed = 0
    sample_num = 50 # for each case
    n = 1
    comet = False
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 2 + 1  # radius [1, 3]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=1000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    n = 2
    comet = False
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 2 + 1  # radius [1, 3]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=1000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    n = 1
    comet = True
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 2 + 1  # radius [1, 3]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=1000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    n = 2
    comet = True
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 2 + 1  # radius [1, 3]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=1000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    all_data = np.array(all_data)
    np.save(f"./data/train.npy", all_data)

def generate_within_data():
    # generate within test sequences
    all_data = []
    seed = 1
    sample_num = 50 # for each case
    n = 1
    comet = False
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 2 + 1  # radius [1, 3]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=1000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    n = 2
    comet = False
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 2 + 1  # radius [1, 3]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=1000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    n = 1
    comet = True
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 2 + 1  # radius [1, 3]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=1000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    n = 2
    comet = True
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 2 + 1  # radius [1, 3]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=1000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    all_data = np.array(all_data)
    np.save(f"./data/within.npy", all_data)

def generate_cross_data():
   # generate cross test sequences
    all_data = []
    seed = 1
    sample_num = 100 # for each case

    n = 8
    comet = False
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 10 + 1  # radius [1, 11]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=3000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    n = 8
    comet = True
    # latin hypercube sampling to generate sample_num*n
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n * sample_num)
    samples = samples.reshape(sample_num, n, 4)
    # scale
    samples[:, :, 0] = samples[:, :, 0] * 2 * np.pi # angle [0, 2pi]
    samples[:, :, 1] = samples[:, :, 1] * (np.pi / 3) - (np.pi / 6) # angle_z [-pi/6, pi/6]
    samples[:, :, 2] = samples[:, :, 2] * 4 + 1  # radius [1, 5]
    samples[:, :, 3] = samples[:, :, 3] * 0.05 + 0.05  # mass [0.05, 0.1]

    actual_sample_num = 0
    for i in range(sample_num):
        data = n_body_simulation(n=n, steps=3000, dt=0.005, save_fps=20, sample=samples[i], comet=comet)
        if data is not None:
            all_data.append(data)
            actual_sample_num += 1
        else:
            print("collision detected in sample", i)
    print(f"actual sample number: {actual_sample_num}")

    all_data = np.array(all_data)
    np.save(f"./data/cross.npy", all_data)

if __name__ == "__main__":
    generate_train_data()
    generate_within_data()
    generate_cross_data()