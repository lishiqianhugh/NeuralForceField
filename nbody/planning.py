import torch
from torch.utils.data import DataLoader
import os
import json
import logging
import importlib
from utils.util import setup_seed
from utils.nbody import NBodyDataset
from configs.nbody_configs import test_arg_parser  # Assuming you have a test argument parser similar to train_arg_parser

# Parse arguments
args = test_arg_parser()

setup_seed(args.seed)

# Setup directories
save_dir = f'./exps/{args.save_dir}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_dir = args.model_path[:args.model_path.rfind('/')]
model_name = f"models.{args.model_name}"
model_module = importlib.import_module(model_name)

# Load the model and its saved configuration
with open(f'{model_dir}/train_args.json', 'r') as f:
    train_args = json.load(f)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if "nff" in model_module.__name__.split(".")[-1]:
    force_predictor = model_module.ForceFieldPredictor(layer_num=train_args['layer_num'], feature_dim=train_args['feature_dim'], hidden_dim=train_args['hidden_dim'])
    ode_func = model_module.ODEFunc(force_predictor)
    model = model_module.NeuralODE(ode_func, train_args['step_size'])
elif "slotformer" in model_module.__name__.split(".")[-1]:
    model = model_module.DynamicsSlotFormer(num_slots=args.num_slots, slot_size=7, history_len=train_args['history_len'], d_model=train_args['hidden_dim'], 
                                            num_layers=train_args['layer_num'], num_heads=4, ffn_dim=256, norm_first=True)
elif "in" in model_module.__name__.split(".")[-1]:
    model = model_module.InteractionNetwork(feature_dim=7, num_layers=train_args['layer_num'], hidden_dim=train_args['hidden_dim'])
else:
    raise ValueError(f"Unknown model name {model_module.__name__}")

# Load trained weights
model.load_state_dict(torch.load(f'{args.model_path}'))
model.to(device)
model.eval()  # Switch to evaluation mode

# Prepare the dataset and data loader
dataset = NBodyDataset(os.path.join('../data/nbody', args.data_path), sample_num=args.sample_num, num_slots=args.num_slots)
test_set = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

# Set up logging
logging.basicConfig(filename=os.path.join(save_dir, 'test.log'), level=logging.INFO)
logging.info("Testing started")

# Evaluate the model
time_steps = args.end_time - args.start_time
time_points = torch.linspace(0, time_steps*0.1, time_steps).to(device)
if 'nff' in model_module.__name__.split(".")[-1]:
    # do backward planning : given the last state, predict the initial state
    time_points = torch.flip(time_points, [0])
    all_loss = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_set):
            # Initial state (t=0)
            initial_state = sample[:, args.end_time-1, :, :].to(device)  # Shape: [bs, body_num, feature_dim]

            # Ground truth trajectory
            true_trajectory = sample[:, args.start_time:args.end_time].to(device)  # Shape: [bs, steps, body_num, feature_dim]

            # Predict trajectory
            pred_trajectory = model(initial_state, time_points)  # Shape: [bs, steps, body_num, feature_dim]

            pred_trajectory = torch.flip(pred_trajectory, [1])
            # Compute loss
            nonzero_mask = true_trajectory[...,0:1] != 0
            batch_mse = torch.mean(
                (pred_trajectory[:, 0:1, :, 1:] * nonzero_mask - true_trajectory[:, 0:1, :, 1:] * nonzero_mask) ** 2,
                dim=(1, 2, 3)
            )
            all_loss.append(batch_mse)
        
        mean_loss = torch.mean(torch.cat(all_loss, dim=0))
        std_loss = torch.std(torch.cat(all_loss, dim=0))
        logging.info(f"Mean Loss: {mean_loss:.4f}, Std Loss: {std_loss:.4f}")
        print(f"Mean Loss: {mean_loss:.4f}, Std Loss: {std_loss:.4f}")
else:
    # use gradient descent to find the initial body properties

    time_points = torch.linspace(0, time_steps*0.1, time_steps).to(device)

    all_loss = []
    for batch_idx, sample in enumerate(test_set):
        # Initial state (t=0)
        initial_state = sample[:, args.start_time, :, :].to(device)  # Shape: [bs, body_num, feature_dim]
        # Ground truth 
        guessing_init_state = torch.nn.Parameter(torch.zeros_like(initial_state[:,:,1:]))
        optimizer = torch.optim.Adam([guessing_init_state], lr=0.1)
        print(f"Initial State: {initial_state[:,:,:]}")
        true_trajectory = sample[:, args.start_time:args.end_time].to(device)  # Shape: [bs, steps, body_num, feature_dim]
        nonzero_mask = true_trajectory[...,0:1] != 0
        for i in range(100):
            optimizer.zero_grad()

            pred_trajectory = model(torch.cat([initial_state[...,:1], guessing_init_state], dim=-1), time_points)
            loss = torch.nn.MSELoss()(pred_trajectory[:,-1:]*nonzero_mask, true_trajectory[:,-1:]*nonzero_mask) # last state
            loss.backward(retain_graph=True)
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {i}, Loss: {loss.item()}")
                nonzero_mask = initial_state[...,0:1] != 0
                batch_guess_mse = torch.mean(
                    (guessing_init_state * nonzero_mask - initial_state[:, :, 1:] * nonzero_mask) ** 2,
                    dim=(1, 2)
                )
                mean_loss = torch.mean(batch_guess_mse)
                std_loss = torch.std(batch_guess_mse)
                logging.info(f"Mean Loss: {mean_loss:.4f}, Std Loss: {std_loss:.4f}")
                print(f"Mean Loss: {mean_loss:.4f}, Std Loss: {std_loss:.4f}")


print("Planning complete.")
