import torch
from torch import nn, optim
import torch.amp
import numpy as np
import time
import os
import shutil
import json
import logging
import importlib
from utils.util import comp_pred_true_traj
from utils.nbody import NBodyDataset
from configs.nbody_configs import train_arg_parser

# Parse arguments
args = train_arg_parser()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = f"models.{args.model_name}"
model_module = importlib.import_module(model_name)

save_dir = f'./exps/{args.save_dir}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

shutil.copy(__file__, save_dir)
shutil.copy(model_module.__file__, save_dir)

with open(f'{save_dir}/train_args.json', 'w') as f:
    json.dump(vars(args), f)


logging.basicConfig(filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.info(f"Training started at {time_str}")

# Load the dataset
dataset = NBodyDataset(os.path.join('../data/nbody', args.data_path), sample_num=args.sample_num, num_slots=args.num_slots)

train_set = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Define model and training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

time_steps = args.end_time - args.start_time
if "nff" in model_module.__name__.split(".")[-1]:
    force_predictor = model_module.ForceFieldPredictor(layer_num=args.layer_num, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim)
    ode_func = model_module.ODEFunc(force_predictor)
    model = model_module.NeuralODE(ode_func, args.step_size,method=args.method,tol = args.tol)
elif "slotformer" in model_module.__name__.split(".")[-1]:
    model = model_module.DynamicsSlotFormer(num_slots=args.num_slots, slot_size=7, history_len=args.history_len, d_model=args.hidden_dim, 
                                            num_layers=args.layer_num, num_heads=4, ffn_dim=256, norm_first=True)
elif "in" in model_module.__name__.split(".")[-1]:
    model = model_module.InteractionNetwork(interaction_feature_dim=args.interaction_feature_dim,num_layers=args.layer_num,hidden_dim=args.hidden_dim)
elif "gcn" in model_module.__name__.split(".")[-1]:
    model = model_module.GCN(feature_dim=args.feature_dim, hidden_dim=args.hidden_dim, num_layers=args.layer_num)
else:
    raise ValueError(f"Unknown model name {model_module.__name__}")

model.to(device)
model.train()

# Training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.minlr)
loss_fn = nn.MSELoss()

SEGMENTS = 10
# Training loop
time_points = torch.linspace(0, time_steps*0.1, time_steps).to(device) # 0.1 is the time interval

assert time_steps % SEGMENTS == 0, "time_steps must be divisible by SEGMENTS"

time_points = time_points[:time_steps//SEGMENTS]

indices = torch.arange(0, time_steps - time_steps//SEGMENTS).unsqueeze(0) + torch.arange(0, time_steps//SEGMENTS).unsqueeze(1)

t1 = time.time()
all_losses = []
for epoch in range(args.num_epochs):
    epoch_loss = 0.0
    for batch_idx, sample in enumerate(train_set):
        sample = sample.to(device)

        sample_stacked = sample[:, args.start_time:args.end_time,:,:][:, indices, :, :]

        true_trajectory = sample_stacked

        sample_stacked = sample_stacked.reshape(-1,time_steps//SEGMENTS, sample.shape[2], sample.shape[3])

        optimizer.zero_grad()
        # Initial state (t=0)
        initial_state = sample_stacked[:, 0, :, :]  # Shape: [bs*stack_num, body_num, feature_dim]

        # Ground truth trajectory
        true_trajectory = true_trajectory.reshape(-1, time_steps//SEGMENTS, sample.shape[2], sample.shape[3])

        # Predict trajectory
        with torch.amp.autocast('cuda'):
            pred_trajectory = model(initial_state, time_points)  # Shape: [bs, steps, body_num, feature_dim]

        # Compute loss
        pred_trajectory = pred_trajectory.reshape(-1, time_steps - time_steps//SEGMENTS, sample.shape[2], sample.shape[3])
        true_trajectory = true_trajectory.reshape(-1, time_steps - time_steps//SEGMENTS, sample.shape[2], sample.shape[3])
        nonzero_mask = true_trajectory[...,0:1] != 0
        pred_trajectory = pred_trajectory * nonzero_mask
        true_trajectory = true_trajectory * nonzero_mask

        loss = loss_fn(pred_trajectory, true_trajectory)

        try:
            scaler = torch.amp.GradScaler('cuda')
        except:
            scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    all_losses.append(epoch_loss / len(train_set))
    if epoch % 5 == 0:
        t2 = time.time()
        print(f"Epoch {epoch}/{args.num_epochs}, Loss: {epoch_loss / len(train_set):.8f}, Time: {t2 - t1:.2f} s")
        logging.info(f"Epoch {epoch}/{args.num_epochs}, Loss: {epoch_loss / len(train_set):.8f}, Time: {t2 - t1:.2f} s")
        t1 = t2
    
    if epoch % args.vis_interval == 0 and epoch > 0:
        # Visualize the trajectory
        comp_pred_true_traj(pred_trajectory.cpu().detach().numpy(), true_trajectory.cpu().detach().numpy(), save_dir=save_dir, label=f'epoch_{epoch}',stride = time_steps//SEGMENTS)
        # vis loss
        import matplotlib.pyplot as plt
        plt.figure()
        # log scale
        plt.plot(range(len(all_losses)), np.log(all_losses))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{save_dir}/loss_{epoch}.png')
        torch.save(model.state_dict(), f'{save_dir}/model_{epoch}.pth')

# save model
torch.save(model.state_dict(), f'{save_dir}/model_final.pth')
print("Training complete.")
