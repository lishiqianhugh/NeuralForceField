import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.debug_draw import vis_trajectory, vis_forces, vis_gif, xyltheta_to_xyxy, vis_force_field

import os
import numpy as np  

from utils.iphyre_data_nff import IPHYREData_seq, pre_process,process_disappear_time
from configs.iphyre_configs import *
from utils.util import *
from utils.evaluation import *

import importlib
import json

args = test_arg_parser()
if args.args_path:
    with open(args.args_path, 'r') as f:
        json_args = json.load(f)

    for key, value in json_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: '{key}' not found in argument list, skipping...")

print(args)

model_name = f"models.{args.model_name}"
model_module = importlib.import_module(model_name)

save_dir = f'./exps/{args.save_dir}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(f'{save_dir}/test_args.json', 'w') as f:
    json.dump(vars(args), f)

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEGMENTS = args.seg

# Initialize model, loss function, and optimizer
if "nff" in model_module.__name__.split(".")[-1]:
    ForceFieldPredictor, ODEFunc, NeuralODEModel = model_module.ForceFieldPredictor, model_module.ODEFunc, model_module.NeuralODEModel
    force_predictor = ForceFieldPredictor(hidden_dim=args.hidden_dim, output_layer=args.layer_num, use_dist_mask=args.use_dist_mask, dist_boundary=0, use_dist_input=args.use_dist_input, angle_scale=args.angle_scale)
    ode_func = ODEFunc(force_predictor, dtheta_scale=args.dtheta_scale)
    model = NeuralODEModel(ode_func, use_adjoint=args.use_adjoint, step_size=args.step_size)
elif "slotformer" in model_module.__name__.split(".")[-1]:
    model = model_module.DynamicsSlotFormer(num_slots=12, slot_size=13, history_len=args.history_len, d_model=args.hidden_dim, 
                                            num_layers=args.layer_num, num_heads=4, ffn_dim=256, norm_first=True, use_dist_mask=args.use_dist_mask)
elif "in" in model_module.__name__.split(".")[-1]:
    model = model_module.InteractionNetwork(history_len=args.history_len,num_layers=args.layer_num,hidden_dim=args.hidden_dim,use_dist_mask=args.use_dist_mask,angle_scale=args.angle_scale)
elif "gcn" in model_module.__name__.split(".")[-1]:
    model = model_module.GCN(history_len=args.history_len,num_layers=args.layer_num,hidden_dim=args.hidden_dim,use_dist_mask=args.use_dist_mask,angle_scale=args.angle_scale)
elif "segno" in model_module.__name__.split(".")[-1]:
    model = model_module.SEGNO(hidden_dim=args.hidden_dim, num_layers=args.layer_num)
else:
    raise ValueError(f"Unknown model name {model_module.__name__}")

test_set = IPHYREData_seq(data_path=f'../data/iphyre/game_seq_data', sample_ids=eval(args.sample), game_ids=eval(args.games))

kwargs = {'pin_memory': True, 'num_workers': 0}
train_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

model.to(device)

# Testing loop
state_dict = torch.load(args.model_path)
if "module." in list(state_dict.keys())[0]:
    from collections import OrderedDict
    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
model.load_state_dict(state_dict)
model.eval()

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
    for batch_idx, (game_names, _, body_property, actions, returns_to_go, timesteps, _,velocity,rotation_angle,angular_velocity) in enumerate(train_loader):
        body_property,actions,returns_to_go,timesteps,velocity,rotation_angle,angular_velocity = body_property.to(device),actions.to(device),returns_to_go.to(device),timesteps.to(device),velocity.to(device),rotation_angle.to(device),angular_velocity.to(device)
        bs, steps, obj_num = body_property.shape[0], body_property.shape[1], 12  # [1, 150, 12, 9] 
        steps = args.end - args.begin
        assert steps % SEGMENTS == 0, "steps must be divisible by SEGMENTS"
        body_property = body_property[:,args.begin:args.end,:,:]
        velocity = velocity[:,args.begin:args.end,:,:]
        rotation_angle = rotation_angle[:,args.begin:args.end,:]
        angular_velocity = angular_velocity[:,args.begin:args.end,:]

        disappear_mask, dynamic_mask, done_mask, body_property, true_trajectories,angular_velocity = \
                        pre_process(body_property,rotation_angle,angular_velocity, bs, steps)
        body_property[:,:,:,3] /= args.angle_scale
        true_trajectories[:,:,:,3] /= args.angle_scale
        angular_velocity /= args.angle_scale

        body_property = body_property.reshape(bs, SEGMENTS, steps//SEGMENTS, obj_num, 9) # [1, 10, 15, 12, 9]
        body_property = body_property.reshape(bs*SEGMENTS, -1 , obj_num, 9) # [10, 15, 12, 9]

        disappear_mask = disappear_mask.reshape(bs,SEGMENTS, steps//SEGMENTS, obj_num) # [1, 10, 15, 12]
        disappear_mask = disappear_mask.reshape(bs*SEGMENTS, -1, obj_num) # [10, 15, 12]

        dynamic_mask = dynamic_mask.reshape(bs,SEGMENTS,  steps//SEGMENTS, obj_num, 1) # [1, 10, 15, 12, 1]
        dynamic_mask = dynamic_mask.reshape(bs*SEGMENTS, -1, obj_num, 1) # [10, 15, 12, 1]

        velocity = velocity.reshape(bs, SEGMENTS, steps//SEGMENTS, obj_num, 2) # [1, 10, 15, 12, 2]
        velocity = velocity.reshape(bs*SEGMENTS, -1, obj_num, 2) # [10, 15, 12, 2]

        angular_velocity = angular_velocity.reshape(bs, SEGMENTS, steps//SEGMENTS, obj_num, 1) # [1, 10, 15, 12, 1]
        angular_velocity = angular_velocity.reshape(bs*SEGMENTS, -1, obj_num, 1) # [10, 15, 12, 1]

        x0 = body_property[:, 0, :, :]
        v0 = velocity[:, 0, :, :]
        angular_velocity = angular_velocity[:, 0, :, :]

        z0 = torch.cat([x0, v0,angular_velocity], dim=-1)  # [args.batch_size*SEGMENTS, obj_num, feature_dim+2]
        t= timesteps[0,:,0]
        t= t[:steps//SEGMENTS] 
        t = t.unsqueeze(0).repeat(bs*SEGMENTS,1)
        disappear_time = process_disappear_time(disappear_mask,steps,SEGMENTS)
        
        predicted_trajectories = model(z0, disappear_time, t)

        predicted_trajectories = predicted_trajectories.reshape(bs, SEGMENTS, steps//SEGMENTS, obj_num, -1) # [1, 10, 15, 12, 9]
        predicted_trajectories = predicted_trajectories.reshape(bs, steps, obj_num, -1) # [1, 150, 12, 9]

        predicted_velocities = predicted_trajectories[:,:,:,FEATURE_DIM:FEATURE_DIM+3] # [1, 150, 12, 3]

        predicted_trajectories = predicted_trajectories[:,:,:,:FEATURE_DIM] # [1, 150, 12, 9]

        predicted_trajectories*= done_mask.unsqueeze(-1).unsqueeze(-1) # [args.batch_size, time_steps, obj_num, feature_dim+2+time_steps]
        true_trajectories*= done_mask.unsqueeze(-1).unsqueeze(-1)
        predicted_velocities*= done_mask.unsqueeze(-1).unsqueeze(-1) # [args.batch_size, time_steps, obj_num, feature_dim+2+time_steps]

        dynamic_mask = dynamic_mask.reshape(bs,steps,obj_num,1) # [1, 150, 12, 1]
        
        for b in range(bs):
            if not args.visualize_force:
                vis_gif(predicted_trajectories[b,:,:,:FEATURE_DIM].reshape(-1,12,9),output_folder=save_dir,save_jpg=False,file_name=f"batch_without_segment_batch={batch_idx}_mini_batch={b}.gif",normalized=True,forces=None,force_idx=None,draw_forces=False,angle_scale=args.angle_scale)
            else:
                dyn_idx = (predicted_trajectories[b,0,:,6].abs() > 0)
                dyn_idx = torch.nonzero(dyn_idx).squeeze(-1)
                num_dyn = dyn_idx.shape[0]
                if "nff" in model_module.__name__.split(".")[-1]:
                    force,_ = force_predictor(init_x=predicted_trajectories[b,:,:,:].reshape(-1,12,9).to(device),
                                                            query_x=predicted_trajectories[b,:,dyn_idx,:].reshape(-1,num_dyn,9).to(device),
                                                            init_v=predicted_velocities[b,:,:,:2].reshape(-1,12,2),
                                                            query_v=predicted_velocities[b,:,dyn_idx,:2].reshape(-1,num_dyn,2),
                                                            init_angular_v=predicted_velocities[b,:,:,-1].reshape(-1,12,1),
                                                            query_angular_v=predicted_velocities[b,:,dyn_idx,2].reshape(-1,num_dyn,1)) # [150,12,len(ball_idx),3]
                
                    force = force.permute(0,2,1,3)[:,:,:,:3] # [150,num_dyn,12,2]
                    mask = torch.eye(12).unsqueeze(0).unsqueeze(-1).to(device)  # [1, 12, 12, 1]
                    mask = mask[:,dyn_idx,:,:] # [1, num_dyn, 12, 1]
                    force = force * (1-mask)

                    gravity = torch.zeros_like(force[:,:,0:1,:]) # [150,num_dyn,1,2]
                    gravity[:,:,0,1] = model.ode_func.gravity.item()
                    force = torch.cat([force,gravity],dim=2) # [150,num_dyn,13,2]
                else:
                    force = torch.zeros((steps,num_dyn,13,2)).to(device)
                vis_gif(predicted_trajectories[b,:,:,:FEATURE_DIM].reshape(-1,12,9),output_folder=save_dir,save_jpg=False,file_name=f"batch_without_segment_batch={batch_idx}_mini_batch={b}.gif",
                                normalized=True,draw_forces=True,forces=force,force_idx=dyn_idx, angle_scale=args.angle_scale)
                vis_gif(true_trajectories[b,:,:,:FEATURE_DIM].reshape(-1,12,9),output_folder=save_dir,save_jpg=False,file_name=f"batch_without_segment_batch={batch_idx}_mini_batch={b}_true.gif",normalized=True,forces=None,force_idx=None,draw_forces=False,angle_scale=args.angle_scale)
        
        # predicted_trajectories_for_draw = xyltheta_to_xyxy(predicted_trajectories, args.angle_scale)
        # true_trajectories_for_draw = xyltheta_to_xyxy(true_trajectories, args.angle_scale)
        # # vis_trajectory(bs,obj_num,dynamic_mask,done_mask,predicted_trajectories_for_draw,true_trajectories_for_draw,save_name=f'{save_dir}/trajectory_batch={batch_idx}',stride=1)
        # if "nff" in model_module.__name__.split(".")[-1]:
        #     vis_forces(bs,obj_num,dynamic_mask,done_mask,predicted_trajectories_for_draw,true_trajectories_for_draw,predicted_trajectories,predicted_velocities,force_predictor,gravity_value=-model.ode_func.gravity.item(),save_name=f'{save_dir}/force_field_batch={batch_idx}',stride=1)
        

        # Evaluate the model
        predict_traj_for_eva = predicted_trajectories[:,:,:,[0,1,3]] # 只对比x,y,theta
        true_traj_for_eva = true_trajectories[:,:,:,[0,1,3]]
        # mask out value beyond 1
        predict_traj_for_eva = torch.clamp(predict_traj_for_eva, min=0, max=1)
        true_traj_for_eva = torch.clamp(true_traj_for_eva, min=0, max=1)
        MAE = MeanAbsoluteError()(predict_traj_for_eva, true_traj_for_eva, dynamic_mask)
        MSE = MeanSquaredError()(predict_traj_for_eva, true_traj_for_eva, dynamic_mask)
        RMSE = RootMeanSquaredError()(predict_traj_for_eva, true_traj_for_eva, dynamic_mask)
        FPE = FinalPositionError()(predict_traj_for_eva, true_traj_for_eva, dynamic_mask)
        PCE = PositionChangeError()(predict_traj_for_eva, true_traj_for_eva, dynamic_mask)
        PCC = PearsonCorrelationCoefficient()(predict_traj_for_eva, true_traj_for_eva, dynamic_mask)
        evaluation_metrics['game_name'].extend(game_names)
        evaluation_metrics['MAE'].extend(MAE.cpu().numpy())
        evaluation_metrics['MSE'].extend(MSE.cpu().numpy())
        evaluation_metrics['RMSE'].extend(RMSE.cpu().numpy())
        evaluation_metrics['FPE'].extend(FPE.cpu().numpy())
        evaluation_metrics['PCE'].extend(PCE.cpu().numpy())
        evaluation_metrics['PCC'].extend(PCC.cpu().numpy())

# save evaluation metrics
df = pd.DataFrame(evaluation_metrics)
df.to_csv(f'{save_dir}/evaluation_metrics.csv', index=False)
print(f"Metrics saved to {save_dir}/evaluation_metrics.csv")
print("Testing complete.")
