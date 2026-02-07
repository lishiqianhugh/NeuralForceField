import rebound
import numpy as np
from scipy.stats.qmc import LatinHypercube
import random

def n_body_simulation(n, steps=3000, dt=0.02, save_fps=300,center_mass_range=None,sample=None, speed_res=0, max_body=9, comet=False):
    """
    通用 n-body 仿真函数
    Args:
        n (int): 天体数量
        steps (int): 仿真步数
        dt (float): 时间步长
        max_body (int): 最大小天体数量，用于padding
    """
    
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.dt = dt

    # 添加中心恒星
    # masses = [1.0, 1.0]
    # xs = [0, 10]
    # ys = [0, 0]
    # zs = [0, 0]
    # vxs = [0, 0]
    # vys = [0, 0]
    # vzs = [0, 0]
    masses = map_uniform_to_ranges(np.random.rand(1), center_mass_range) # 恒星质量范围
    masses = list(masses)

    xs = [0]
    ys = [0]
    zs = [0]
    vxs = [0]
    vys = [0]
    vzs = [0]
    for i in range(len(masses)):
        sim.add(m=masses[i], x=xs[i], y=ys[i], z=zs[i], vx=vxs[i], vy=vys[i], vz=vzs[i])

    # 添加围绕恒星运行的行星
    for i in range(n):
        angle_xy = sample[i, 0]  # xy平面的角度
        angle_z = sample[i, 1]   # z方向倾角，可以调整范围
        radius = sample[i, 2] 
        
        x = radius * np.cos(angle_z) * np.cos(angle_xy)
        y = radius * np.cos(angle_z) * np.sin(angle_xy)
        z = radius * np.sin(angle_z)
        if comet:
            speed = np.sqrt(2*sim.particles[0].m / radius)
        else:
            speed = np.sqrt(sim.particles[0].m / radius)
        
        # 计算速度分量，保证垂直于径向
        vx = -speed * np.cos(angle_z) * np.sin(angle_xy)
        vy = speed * np.cos(angle_z) * np.cos(angle_xy)
        vz = 0  # 初始垂直速度设为0

        mass = sample[i, 3]
        masses.append(mass)
        sim.add(m=mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)


    # 记录轨迹
    traj_masses = [[] for _ in range(n + 1)]
    positions = [[] for _ in range(n + 1)]
    velocities = [[] for _ in range(n + 1)]
    accs = [[] for _ in range(n + 1)]
    for t in range(steps):
        # 检测碰撞
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

    # 保存物体属性到.npy 包括质量，位置，速度
    data = np.concatenate([np.array(traj_masses), np.array(positions), np.array(velocities), np.array(accs)], axis=-1) # [body_num, steps, 10]
    # downsampling steps to 100
    data = data[:, ::save_fps, :]
    data = data.transpose(1, 0, 2)  # [steps, body_num, 10]
    # 填充为 max_body 数量
    if n < max_body:
        padding_shape = (data.shape[0], max_body - n, data.shape[2])
        padding = np.zeros(padding_shape)
        data = np.concatenate([data, padding], axis=1)
    # vis_nbody_traj(data)

    return data

def map_uniform_to_ranges(x, ranges):
    """
    Maps values from a uniform distribution [0, 1] to a set of specified ranges,
    maintaining uniformity across the union of these ranges.

    Args:
        x (np.ndarray): Input array with values uniformly distributed in [0, 1].
        ranges (list): A list of tuples, where each tuple (min_val, max_val)
                       defines a range.

    Returns:
        np.ndarray: Array with values mapped to the specified ranges.
    """
    ranges = np.array(ranges)
    lengths = ranges[:, 1] - ranges[:, 0]
    total_length = np.sum(lengths)

    if total_length <= 0:
        raise ValueError("Total length of ranges must be positive.")

    cum_lengths = np.insert(np.cumsum(lengths), 0, 0)

    scaled_x = x * total_length

    indices = np.clip(np.searchsorted(cum_lengths, scaled_x, side='right') - 1, 0, len(ranges) - 1)
    offset = scaled_x - cum_lengths[indices]
    range_starts = ranges[indices, 0]
    mapped_x = range_starts + offset

    return mapped_x

def generate_data(params,save_path):
    # generate training sequences
    all_data = []
    np.random.seed(params['seed'])
    random.seed(params['seed'])
    
    for sample_num, planet_num, whether_comet in params['cases']:
        print(f"sample_num: {sample_num}, planet_num: {planet_num}, whether_comet: {whether_comet}")
        comet = whether_comet
        # latin hypercube sampling to generate sample_num*n
        sampler = LatinHypercube(d=4, seed=params['seed'])
        samples = sampler.random(planet_num * 1000)
        samples = samples.reshape(1000, planet_num, 4) # [sample_num, n, 4] in range [0, 1]

        # scale
        samples[:, :, 0] = map_uniform_to_ranges(samples[:, :, 0], params['angle_ranges']) # angle [0, 2pi]
        samples[:, :, 1] = map_uniform_to_ranges(samples[:, :, 1], params['angle_z_ranges']) # angle_z [-pi/6, pi/6]
        samples[:, :, 2] = map_uniform_to_ranges(samples[:, :, 2], params['radius_ranges'])  # radius [1, 3]
        samples[:, :, 3] = map_uniform_to_ranges(samples[:, :, 3], params['mass_ranges'])  # mass [0.05, 0.1]

        actual_sample_num = 0
        for i in range(1000):
            data = n_body_simulation(n=planet_num, steps=params['steps'], dt=0.005, save_fps=20,center_mass_range=params['center_mass_ranges'],sample=samples[i], comet=comet,max_body=params['max_body'])
            if data is not None:
                all_data.append(data)
                actual_sample_num += 1
                if actual_sample_num == sample_num:
                    break
            else:
                print("collision detected in sample", i)

        print(f"actual sample number: {actual_sample_num}")

    all_data = np.array(all_data)
    np.save(save_path, all_data)

train_params = {
    'steps': 1000,
    'angle_ranges': [(0, 2 * np.pi)],
    'max_body': 2,
    'angle_z_ranges': [(-np.pi / 6, np.pi / 6)],
    'radius_ranges': [(1, 3)],
    'mass_ranges': [(0.05, 0.1)],
    'center_mass_ranges': [(3, 5),(7,9)],
    'seed': 0,
    'cases': [(50,1,False),(50,1,True),(50,2,False),(50,2,True)]
}

train_long_params = {
    'steps': 1000,
    'angle_ranges': [(0, 2 * np.pi)],
    'max_body': 9,
    'angle_z_ranges': [(-np.pi / 6, np.pi / 6)],
    'radius_ranges': [(1, 3)],
    'mass_ranges': [(0.05, 0.1)],
    'center_mass_ranges': [(3, 5),(7,9)],
    'seed': 0,
    'cases': [(50,1,False),(50,1,True),(50,2,False),(50,2,True)]
}

within_params = {
    'steps': 3000,
    'angle_ranges': [(0, 2 * np.pi)],
    'max_body': 2,
    'angle_z_ranges': [(-np.pi / 6, np.pi / 6)],
    'radius_ranges': [(1, 3)],
    'mass_ranges': [(0.05, 0.1)],
    'center_mass_ranges': [(3, 5),(7,9)],
    'seed': 1,
    'cases': [(50,1,False),(50,1,True),(50,2,False),(50,2,True)]
}

within_long_params = {
    'steps': 3000,
    'angle_ranges': [(0, 2 * np.pi)],
    'max_body': 9,
    'angle_z_ranges': [(-np.pi / 6, np.pi / 6)],
    'radius_ranges': [(1, 3)],
    'mass_ranges': [(0.05, 0.1)],
    'center_mass_ranges': [(3, 5),(7,9)],
    'seed': 1,
    'cases': [(50,1,False),(50,1,True),(50,2,False),(50,2,True)]
}

cross_params = {
    'steps': 3000,
    'angle_ranges': [(0, 2 * np.pi)],
    'max_body': 9,
    'angle_z_ranges': [(-np.pi / 6, np.pi / 6)],
    'radius_ranges': [(1, 5)],
    'mass_ranges': [(0.05, 0.15)],
    'center_mass_ranges': [(3, 9)],
    'seed': 2,
    'cases': [(50,7,False),(50,7,True),(50,9,False),(50,9,True)]
}


if __name__ == "__main__":
    print("Generating training data...")
    generate_data(train_params, '../data/nbody/train_data.npy')
    print("Generating training data long...")
    generate_data(train_long_params, '../data/nbody/train_data_long.npy')
    print("Generating within data...")
    generate_data(within_params, '../data/nbody/within_data.npy')
    print("Generating within data long...")
    generate_data(within_long_params, '../data/nbody/within_data_long.npy')
    print("Generating cross data...")
    generate_data(cross_params, '../data/nbody/cross_data.npy')
