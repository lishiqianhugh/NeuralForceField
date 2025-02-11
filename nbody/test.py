import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import shutil
import json
import logging
import importlib
import pandas as pd
from utils.util import setup_seed
from utils.evaluation import *
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

evaluation_metrics = {
    'game_name':[],
    'MAE': [],
    'MSE': [],
    'RMSE': [],
    'FPE': [],
    'PCE': [],
    'PCC': []
}

with torch.no_grad():
    total_loss = 0.0
    for batch_idx, sample in enumerate(test_set):
        # Initial state (t=0)
        initial_state = sample[:, args.start_time, :, :].to(device)  # Shape: [bs, body_num, feature_dim]
        # Ground truth trajectory
        true_trajectory = sample[:, args.start_time:args.end_time].to(device)  # Shape: [bs, steps, body_num, feature_dim]
        # Predict trajectory
        pred_trajectory = model(initial_state, time_points)  # Shape: [bs, steps, body_num, feature_dim]
        nonzero_mask = true_trajectory[...,0:1] != 0
        # Compute loss
        loss = torch.nn.MSELoss()(pred_trajectory[...,:4]*nonzero_mask, true_trajectory[...,:4]*nonzero_mask)
        total_loss += loss.item()

        # Evaluate the model
        predict_traj_for_eva = pred_trajectory
        true_traj_for_eva = true_trajectory
        MAE = MeanAbsoluteError()(predict_traj_for_eva, true_traj_for_eva, nonzero_mask)
        MSE = MeanSquaredError()(predict_traj_for_eva, true_traj_for_eva, nonzero_mask)
        RMSE = RootMeanSquaredError()(predict_traj_for_eva, true_traj_for_eva, nonzero_mask)
        FPE = FinalPositionError()(predict_traj_for_eva, true_traj_for_eva, nonzero_mask)
        PCE = PositionChangeError()(predict_traj_for_eva, true_traj_for_eva, nonzero_mask)
        PCC = PearsonCorrelationCoefficient()(predict_traj_for_eva, true_traj_for_eva, nonzero_mask)
        evaluation_metrics['game_name'].extend(range(len(MAE)))
        evaluation_metrics['MAE'].extend(MAE.cpu().numpy())
        evaluation_metrics['MSE'].extend(MSE.cpu().numpy())
        evaluation_metrics['RMSE'].extend(RMSE.cpu().numpy())
        evaluation_metrics['FPE'].extend(FPE.cpu().numpy())
        evaluation_metrics['PCE'].extend(PCE.cpu().numpy())
        evaluation_metrics['PCC'].extend(PCC.cpu().numpy())
    
    avg_loss = total_loss / len(test_set)
    logging.info(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Forward Average Test Loss: {avg_loss:.6f}")

# save evaluation metrics
df = pd.DataFrame(evaluation_metrics)
df.to_csv(f'{save_dir}/{args.data_path.split(".")[0]}.csv', index=False)
print(f"Metrics saved to {save_dir}/{args.data_path.split('.')[0]}.csv")
# comp_pred_true_traj(pred_trajectory.cpu().detach().numpy(), true_trajectory.cpu().detach().numpy(), save_dir=save_dir, label='test')
print("Testing complete.")
