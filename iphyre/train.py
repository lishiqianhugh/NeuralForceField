import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import time
import numpy as np  
import logging
import shutil

from configs.iphyre_configs import *
from utils.iphyre_data_nff import IPHYREData_seq, pre_process,process_disappear_time,process_stacking_data
from utils.util import *
from utils.loss import WeightedMSELoss
from utils.debug_draw import vis_trajectory, vis_forces, xyltheta_to_xyxy,xyxy_to_xyltheta,draw_heatmap
import importlib
import json

args = train_arg_parser()
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

torch.manual_seed(args.seed)
np.random.seed(args.seed)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH_SEGMENTS = eval(args.segments)
    
# Initialize model, loss function, and optimizer
if "nff" in model_module.__name__.split(".")[-1]:
    ForceFieldPredictor, ODEFunc, NeuralODEModel = model_module.ForceFieldPredictor, model_module.ODEFunc, model_module.NeuralODEModel
    force_predictor = ForceFieldPredictor(hidden_dim=args.hidden_dim, output_layer=args.layer_num, use_dist_mask=args.use_dist_mask, dist_boundary=0, use_dist_input=args.use_dist_input, angle_scale=args.angle_scale)
    ode_func = ODEFunc(force_predictor, dtheta_scale=args.dtheta_scale)
    model = NeuralODEModel(ode_func, use_adjoint=args.use_adjoint, step_size=args.step_size)
elif "slotformer" in model_module.__name__.split(".")[-1]:
    model = model_module.DynamicsSlotFormer(num_slots=12, slot_size=13, history_len=args.history_len, d_model=args.hidden_dim, 
                                            num_layers=args.layer_num, num_heads=4, ffn_dim=256, norm_first=True, 
                                            slotres_scale=args.slotres_scale, use_dist_mask=args.use_dist_mask)
elif "in" in model_module.__name__.split(".")[-1]:
    model = model_module.InteractionNetwork(history_len=args.history_len,num_layers=args.layer_num,hidden_dim=args.hidden_dim,use_dist_mask=args.use_dist_mask,angle_scale=args.angle_scale)
elif "gcn" in model_module.__name__.split(".")[-1]:
    model = model_module.GCN(history_len=args.history_len,num_layers=args.layer_num,hidden_dim=args.hidden_dim,use_dist_mask=args.use_dist_mask,angle_scale=args.angle_scale)
else:
    raise ValueError(f"Unknown model name {model_module.__name__}")

criterion = WeightedMSELoss(gamma=args.gamma, alpha=args.alpha, beta=args.beta)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.minlr)

train_set = IPHYREData_seq(data_path=f'../data/iphyre/game_seq_data', sample_ids=eval(args.sample), game_ids=eval(args.games))
kwargs = {'pin_memory': True, 'num_workers': 0}
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

model.to(device)
model = nn.DataParallel(model)
try:
    scaler = torch.amp.GradScaler('cuda')
except:
    scaler = torch.cuda.amp.GradScaler()

model.train()
t1 = time.time()
minibatch = 0
for epoch in range(args.num_epochs):
    epoch_loss, epoch_mse, epoch_res, epoch_res_res = 0, 0, 0, 0
    SEGMENTS = EPOCH_SEGMENTS[min(epoch//(args.num_epochs//len(EPOCH_SEGMENTS)),len(EPOCH_SEGMENTS)-1)]
    for batch_idx, (_, _, body_property, actions, returns_to_go, timesteps, _,velocity,rotation_angle,angular_velocity) in enumerate(train_loader):
        body_property,actions,returns_to_go,timesteps,velocity,rotation_angle,angular_velocity = body_property.to(device),actions.to(device),returns_to_go.to(device),timesteps.to(device),velocity.to(device),rotation_angle.to(device),angular_velocity.to(device)

        optimizer.zero_grad()
        minibatch += 1

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

        stack_num = steps//SEGMENTS
        body_property_stacked, disappear_mask_stacked, velocity_stacked, angular_velocity_stacked, true_trajectories_stacked, done_mask_stacked = \
                        process_stacking_data(body_property,true_trajectories,disappear_mask,velocity,angular_velocity,done_mask,bs,steps,obj_num,SEGMENTS,stack_num)
        
        x0 = body_property_stacked[:, 0, :, :] 
        v0 = velocity_stacked[:, 0, :, :] 
        angular_velocity = angular_velocity_stacked[:, 0, :, :]

        z0 = torch.cat([x0, v0,angular_velocity], dim=-1)  # Initial state [bs*SEGMENTS, obj_num, feature_dim+2]
        t= timesteps[0,:steps,0]
        t= t[:steps//SEGMENTS]
        t = t.unsqueeze(0).repeat(bs*stack_num*(SEGMENTS-1),1) # [10, 15]

        disappear_time_stacked = process_disappear_time(disappear_mask_stacked, steps, SEGMENTS)
        try:
            with torch.amp.autocast('cuda'):
                predicted_trajectories = model(z0=z0, disappear_time=disappear_time_stacked, t=t)
        except:
            with torch.cuda.amp.autocast():
                predicted_trajectories = model(z0=z0, disappear_time=disappear_time_stacked, t=t)
        predicted_trajectories = predicted_trajectories.reshape(stack_num*bs, steps//SEGMENTS*(SEGMENTS-1), obj_num, -1)
        true_trajectories_stacked = true_trajectories_stacked.reshape(stack_num*bs, steps//SEGMENTS*(SEGMENTS-1), obj_num, -1)
        done_mask_stacked = done_mask_stacked.reshape(stack_num*bs, steps//SEGMENTS*(SEGMENTS-1)).unsqueeze(-1).unsqueeze(-1)

        predicted_velocities = predicted_trajectories[:,:,:,FEATURE_DIM:FEATURE_DIM+3] # [1, 150, 12, 3]

        predicted_trajectories*= done_mask_stacked # [bs, time_steps, obj_num, feature_dim+2+time_steps]
        true_trajectories_stacked*= done_mask_stacked # [bs, time_steps, obj_num, feature_dim+2+time_steps]
        predicted_velocities*= done_mask_stacked # [bs, time_steps, obj_num, feature_dim+2+time_steps]

        loss, mse, res_loss, res_res_loss = criterion(predicted_trajectories[:,:,:,:4], true_trajectories_stacked[:,:,:,:4])

        if "nff_model_direct" in model_module.__name__.split(".")[-1]:
            direction_aux_loss = nn.MSELoss()(predicted_trajectories[:,:,:,-1],torch.zeros_like(predicted_trajectories[:,:,:,-1]))
            loss += direction_aux_loss

        scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        epoch_loss += loss
        epoch_mse += mse
        epoch_res += res_loss
        epoch_res_res += res_res_loss
        
        if (epoch % args.vis_interval == 0 and epoch != 0):
            predicted_trajectories_for_draw = xyltheta_to_xyxy(predicted_trajectories[:,:,:,:4], args.angle_scale)
            true_trajectories_for_draw = xyltheta_to_xyxy(true_trajectories_stacked, args.angle_scale)

            vis_trajectory(bs,obj_num,dynamic_mask,done_mask,predicted_trajectories_for_draw,true_trajectories_for_draw,save_name=f'{save_dir}/trajectory_batch={batch_idx}_epoch={epoch}',stride=steps//SEGMENTS)
            if "nff" in model_module.__name__.split(".")[-1]:
                vis_forces(bs,obj_num,dynamic_mask,done_mask,predicted_trajectories_for_draw,true_trajectories_for_draw,predicted_trajectories,predicted_velocities,force_predictor,gravity_value=-model.module.ode_func.gravity.item(),save_name=f'{save_dir}/force_batch={batch_idx}_epoch={epoch}',stride=steps//SEGMENTS)

        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

    scheduler.step()

    epoch_loss /= len(train_loader)
    epoch_mse /= len(train_loader)
    epoch_res /= len(train_loader)
    epoch_res_res /= len(train_loader)

    if epoch % 5 == 0:
        t2 = time.time()
        if "nff_model_direct" in model_module.__name__.split(".")[-1]:
            print(f'Epoch [{epoch}/{args.num_epochs}], SEGMENTS:{SEGMENTS}, Loss: {epoch_loss.item()}, MSE: {epoch_mse.item()}, Residual Loss: {epoch_res.item()}, Residual Residual Loss: {epoch_res_res.item()}, Direction Aux Loss: {direction_aux_loss.item()}, Time: {t2-t1}')
            logging.info(f'Epoch [{epoch}/{args.num_epochs}], SEGMENTS:{SEGMENTS}, Loss: {epoch_loss.item()}, MSE: {epoch_mse.item()}, Residual Loss: {epoch_res.item()}, Residual Residual Loss: {epoch_res_res.item()}, Direction Aux Loss: {direction_aux_loss.item()}, Time: {t2-t1}')
        else:
            print(f'Epoch [{epoch}/{args.num_epochs}], SEGMENTS:{SEGMENTS}, Loss: {epoch_loss.item()}, MSE: {epoch_mse.item()}, Residual Loss: {epoch_res.item()}, Residual Residual Loss: {epoch_res_res.item()}, Time: {t2-t1}')
            logging.info(f'Epoch [{epoch}/{args.num_epochs}], SEGMENTS:{SEGMENTS}, Loss: {epoch_loss.item()}, MSE: {epoch_mse.item()}, Residual Loss: {epoch_res.item()}, Residual Residual Loss: {epoch_res_res.item()}, Time: {t2-t1}')
        t1 = t2

    if epoch % 100 == 0 and epoch != 0:
        vis_losscurve(epoch,f"{save_dir}/train.log")

    if epoch % args.model_interval == 0 and epoch != 0:     
        torch.save(model.state_dict(), f'{save_dir}/model_{epoch}.pt')
    
print("Training complete.")
torch.save(model.state_dict(), f'{save_dir}/model_final.pt')
time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.info(f"Training ended at {time_str}")
