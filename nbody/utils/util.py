import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap
import os
import torch
import random
import numpy as np
import pandas as pd

def draw_heatmap(path_a, path_b, save_path):
    df1 = pd.read_csv(path_a)
    df2 = pd.read_csv(path_b)

    array1 = df1.to_numpy()
    array2 = df2.to_numpy()

    if array1.shape != array2.shape:
        raise ValueError("The two CSV files do not have the same shape.")

    array_diff = array1 - array2

    plt.imshow(array_diff, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'{save_path}/heatmap.png')
    np.savetxt(f'{save_path}/difference.csv', array_diff, delimiter=',', fmt='%f')
    print(f"The difference has been saved to '{save_path}/difference.csv'.")
    plt.close()
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def vis_nbody_traj(data, save_dir=None, label='debug'):
    # visualize the data 3d
    if len(data.shape) == 3:
        data = data[None, ...]
    for bs in range(data.shape[0]):
        n = data.shape[2] - 1 
        positions = data[bs,:,:,1:4]  # [bs, steps, body_num, 3]
        masses = data[bs, 0, :, 0]  # [body_num]
        final_x = positions[-1, :, 0]
        final_y = positions[-1, :, 1]
        final_z = positions[-1, :, 2]

        cmap = get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(n + 1)]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(n + 1):
            x_vals = [pos[0] for pos in positions[:,i]]
            y_vals = [pos[1] for pos in positions[:,i]]
            z_vals = [pos[2] for pos in positions[:,i]]
            ax.plot(x_vals, y_vals, z_vals, color=colors[i], label=f"Body {i}" if i > 0 else "Star", alpha=0.6, linewidth=2.5)

        for i in range(n + 1):
            ax.scatter(final_x[i], final_y[i], final_z[i], 
                        s=masses[i] * 200,
                        color=colors[i], alpha=0.8, label=f"Body {i}" if i > 0 else "Star")

        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_zlabel("Z [AU]")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        # plt.legend()
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/{label}_nbody_traj_{bs}.png", dpi=300)
        else:
            plt.show()
        
        plt.close()

def comp_pred_true_traj(pred, true, save_dir=None, label='train',stride=1):
    '''
    pred: [bs, steps, body_num, 10]
    true: [bs, steps, body_num, 10]
    stride: the stride of the time steps
    visualize the predicted and true trajectory in one plot
    pred_traj: solid line
    true_traj: dashed line
    '''
    # visualize the data 3d
    if len(pred.shape) == 3:
        pred = pred[None, ...]
        true = true[None, ...]
    for bs in range(0,pred.shape[0],stride):
        n = pred.shape[2] - 1
        pred_positions = pred[bs,:,:,1:4]
        true_positions = true[bs,:,:,1:4]
        masses = pred[bs, 0, :, 0]
        final_x = pred_positions[-1, :, 0]
        final_y = pred_positions[-1, :, 1]
        final_z = pred_positions[-1, :, 2]

        cmap = get_cmap("tab10")  
        colors = [cmap(i % 10) for i in range(n + 1)]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n + 1):
            x_vals = [pos[0] for pos in pred_positions[:,i]]
            y_vals = [pos[1] for pos in pred_positions[:,i]]
            z_vals = [pos[2] for pos in pred_positions[:,i]]
            ax.plot(x_vals, y_vals, z_vals, color=colors[i], label=f"Body {i}" if i > 0 else "Star", alpha=0.6)
            ax.scatter(final_x[i], final_y[i], final_z[i], 
                        s=masses[i] * 200, 
                        color=colors[i], alpha=0.8, label=f"Body {i}" if i > 0 else "Star")
        for i in range(n + 1):
            x_vals = [pos[0] for pos in true_positions[:,i]]
            y_vals = [pos[1] for pos in true_positions[:,i]]
            z_vals = [pos[2] for pos in true_positions[:,i]]
            ax.plot(x_vals, y_vals, z_vals, color=colors[i], linestyle='--', alpha=0.6)
            

        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_zlabel("Z [AU]")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        plt.legend()
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/{label}_nbody_traj_{bs}.png", dpi=300)
        else:
            plt.show()
        
        plt.close()

def save_tensor_as_csv(tensor, file_path):
    
    newtensor = tensor.cpu()
    if len(tensor.shape) > 2:
        newtensor = newtensor.reshape(-1, tensor.shape[-1])

    newtensor = newtensor.detach().numpy()
    if not isinstance(tensor, np.ndarray):
        newtensor = np.array(newtensor)
    
    
    df = pd.DataFrame(newtensor)
    df.to_csv(file_path, index=False, header=False)


if __name__ == '__main__':
    # test vis_nbody_traj
    import numpy as np
    data = np.load('nbody_nocollision.npy', allow_pickle=True)
    vis_nbody_traj(data)