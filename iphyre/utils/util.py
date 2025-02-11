import torch
import numpy as np
import random
import csv
import pandas as pd
import re
import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_tensor_as_csv(tensor, file_path):
    
    newtensor = tensor.cpu()
    if len(tensor.shape) > 2:
        newtensor = newtensor.reshape(-1, tensor.shape[-1])

    newtensor = newtensor.detach().numpy()
    if not isinstance(tensor, np.ndarray):
        newtensor = np.array(newtensor)
    
    
    df = pd.DataFrame(newtensor)
    
    df.to_csv(file_path, index=False, header=False)

def calculate_distribution(data):

    logs = np.log10(np.abs(data))
    bins = np.floor(logs)
    bin_counts = {}
    for bin in bins:
        if bin in bin_counts:
            bin_counts[bin] += 1
        else:
            bin_counts[bin] = 1
    
    total_count = len(data)
    bin_counts = dict(sorted(bin_counts.items(), key=lambda x: x[0]))
    for bin, count in bin_counts.items():
        print(f'10^{bin} - 10^{bin+1}: {count/total_count:.4f}')


def vis_losscurve(steps, log_file):
    epochs = []
    losses = []
    mse_losses = []
    residual_losses = []
    residual_residual_losses = []

    # Updated log pattern to capture all loss values
    log_pattern = re.compile(r"Epoch\s*\[(\d+)/\d+\].*Loss:\s*([\d\.eE\-]+)\s*,\s*MSE:\s*([\d\.eE\-]+)\s*,\s*Residual\s*Loss:\s*([\d\.eE\-]+)\s*,\s*Residual\s*Residual\s*Loss:\s*([\d\.eE\-]+)")
    # log_pattern = re.compile(r"Epoch\s*\[(\d+)/\d+\].*Loss:\s*([\d\.eE\-]+)\s*,\s*Time:\s*([\d\.eE\-]+)")

    with open(log_file, 'r') as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                mse_loss = float(match.group(3))
                residual_loss = float(match.group(4))
                residual_residual_loss = float(match.group(5))

                # residual_loss = float(match.group(3))
                # residual_residual_loss = float(match.group(3))
                
                epochs.append(epoch)
                losses.append(loss)
                mse_losses.append(mse_loss)
                residual_losses.append(residual_loss)
                residual_residual_losses.append(residual_residual_loss)

    # Apply log transformation to the losses
    losses = np.log(losses)
    mse_losses = np.log(mse_losses)
    residual_losses = np.log(residual_losses)
    residual_residual_losses = np.log(residual_residual_losses)

    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label='Loss')
    plt.plot(epochs, mse_losses, label='MSE')
    plt.plot(epochs, residual_losses, label='Residual Loss')
    plt.plot(epochs, residual_residual_losses, label='Residual Residual Loss')
    
    # Labels and Title
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title(f'Loss Curve for {steps} steps (log scale)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    log_file_path = log_file[:log_file.rfind('/')+1]
    output_file = f'{log_file_path}loss_curve_{steps}.png'
    print(f'Saving plot to {output_file}')
    plt.savefig(output_file)
    plt.close()
