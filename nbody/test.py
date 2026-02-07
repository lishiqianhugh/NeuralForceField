import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import shutil
import json
import logging
import importlib
import pandas as pd
from utils.util import vis_nbody_traj, comp_pred_true_traj, setup_seed, save_tensor_as_csv, draw_heatmap
from utils.evaluation import *
from utils.nbody import NBodyDataset
from configs.nbody_configs import test_arg_parser  # Assuming you have a test argument parser similar to train_arg_parser

# Parse arguments
args = test_arg_parser()

setup_seed(args.seed)

# Setup directories
save_dir = f'./results/{args.save_dir}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
model_dir = os.path.dirname(args.model_path)
model_name = f"models.{args.model_name}"
model_module = importlib.import_module(model_name)

# Load the model and its saved configuration
with open(f'{model_dir}/train_args.json', 'r') as f:
    train_args = json.load(f)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.feature_dim = train_args['feature_dim']
args.hidden_dim = train_args['hidden_dim']
args.layer_num = train_args['layer_num']
args.num_slots = train_args['num_slots']
args.history_len = train_args['history_len']
args.interaction_feature_dim = train_args['interaction_feature_dim']
args.in_rollout = train_args['in_rollout']

if model_module.__name__.split(".")[-1]=="nff":
    force_predictor = model_module.ForceFieldPredictor(layer_num=args.layer_num, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim)
    ode_func = model_module.ODEFunc(force_predictor)
    model = model_module.NeuralODE(ode_func, args.step_size,method=args.method,tol = args.tol)
elif model_module.__name__.split(".")[-1]=="slotformer":
    model = model_module.DynamicsSlotFormer(num_slots=args.num_slots, slot_size=7, history_len=args.history_len, d_model=args.hidden_dim, 
                                            num_layers=args.layer_num, num_heads=4, ffn_dim=256, norm_first=True)
elif model_module.__name__.split(".")[-1]=="in":
    model = model_module.InteractionNetwork(interaction_feature_dim=args.interaction_feature_dim,num_layers=args.layer_num,hidden_dim=args.hidden_dim)
elif model_module.__name__.split(".")[-1]=="segno" or model_module.__name__.split(".")[-1]=="segno_new":
    model = model_module.SEGNO(hidden_dim=args.hidden_dim, num_layers=args.layer_num)
else:
    raise ValueError(f"Unknown model name {model_module.__name__}")


# Load trained weights
model.load_state_dict(torch.load(f'{args.model_path}'))
model.to(device)
model.eval()

# Prepare the dataset and data loader
dataset = NBodyDataset(os.path.join('../data/nbody', args.data_path), sample_num=args.sample_num)
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

        data = true_trajectory[0,:,:2,:].cpu().detach().numpy()

        # Predict trajectory
        pred_trajectory = model(initial_state, time_points)  # Shape: [bs, steps, body_num, feature_dim]

        nonzero_mask = true_trajectory[...,0:1] != 0 # [bs, steps, body_num, 1]
        
        # Create force mask where both bodies need to be non-zero
        force_mask = nonzero_mask.unsqueeze(3) & nonzero_mask.unsqueeze(2) # [bs, steps, body_num, body_num, 1]
        pred_trajectory = pred_trajectory * nonzero_mask

        loss = torch.nn.MSELoss()(pred_trajectory[...,:]*nonzero_mask, true_trajectory[...,:]*nonzero_mask)
        total_loss += loss.item()

        comp_pred_true_traj(pred_trajectory[:1].cpu().detach().numpy(), true_trajectory[:1].cpu().detach().numpy(), save_dir=save_dir, label='test')

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