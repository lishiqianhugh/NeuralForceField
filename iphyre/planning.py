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
from utils.iphyre_data_nff import IPHYREData_seq, pre_process,process_disappear_time,process_stacking_data,process_joint_length
from utils.util import *
from utils.loss import WeightedMSELoss
from utils.debug_draw import vis_trajectory, vis_forces, xyltheta_to_xyxy,xyxy_to_xyltheta,draw_heatmap,vis_gif
import importlib

from iphyre.simulator import IPHYRE
from evaluation import evaluate_model
import json

args = planning_arg_parser()
if args.args_path:
    with open(args.args_path, 'r') as f:
        json_args = json.load(f)

    for key, value in json_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: '{key}' not found in argument list, skipping...")

print(args)

model_name = f"{args.model_path.split('/')[0]}.{args.model_path.split('/')[1]}.{args.model_name}"
print(model_name)
model_module = importlib.import_module(model_name)

setup_seed(args.seed)

save_dir = f'./exps/{args.save_dir}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

shutil.copy(__file__, save_dir)
shutil.copy(model_module.__file__, save_dir)

with open(f'{save_dir}/planning_args.json', 'w') as f:
    json.dump(vars(args), f)

logging.basicConfig(filename=os.path.join(save_dir, 'refine.log'), level=logging.INFO)
time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.info(f"Planning & refining started at {time_str}")

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
time_steps = args.time_steps

# settings for the dataset
obj_num = 12

# Initialize model, loss function, and optimizer
if "nff" in model_module.__name__.split(".")[-1]:
    ForceFieldPredictor, ODEFunc, NeuralODEModel = model_module.ForceFieldPredictor, model_module.ODEFunc, model_module.NeuralODEModel
    force_predictor = ForceFieldPredictor(hidden_dim=args.hidden_dim, output_layer=args.layer_num, use_dist_mask=args.use_dist_mask, dist_boundary=0, use_dist_input=args.use_dist_input, angle_scale=args.angle_scale)
    ode_func = ODEFunc(force_predictor, dtheta_scale=args.dtheta_scale,acceleration_clip=args.acceleration_clip)
    model = NeuralODEModel(ode_func, use_adjoint=args.use_adjoint, step_size=args.step_size)
elif "slotformer" in model_module.__name__.split(".")[-1]:
    model = model_module.DynamicsSlotFormer(num_slots=12, slot_size=13, history_len=args.history_len, d_model=args.hidden_dim, 
                                            num_layers=args.layer_num, num_heads=4, ffn_dim=256, norm_first=True, 
                                            slotres_scale=args.slotres_scale, use_dist_mask=args.use_dist_mask)
elif "in" in model_module.__name__.split(".")[-1]:
    model = model_module.InteractionNetwork(history_len=args.history_len,num_layers=args.layer_num,hidden_dim=args.hidden_dim,use_dist_mask=args.use_dist_mask,angle_scale=args.angle_scale)
else:
    raise ValueError(f"Unknown model name {model_module.__name__}")

def generate_disappear_time(eli_obj_idx, trial_id, max_game_time=15, max_action_time=7, interval=0.1, epsilon=0.2):
    """
    Input: 
    eli_obj_idx [bs, obj_num] represents whether a certain object can disappear
    device: the device (e.g. 'cuda' or 'cpu')
    max_game_time: maximum game time
    max_action_game: the longest time during which actions can be performed
    epsilon: each object has an epsilon probability of not disappearing
    
    Output: 
    disappear_time [bs, obj_num, 1] represents the time at which each object disappears
    If the object cannot disappear, its disappear time = (time_steps+1)/10
    If the object can disappear, a random time from [0, time_steps] is chosen, then divided by 10
    Time cannot be duplicated, except for 0 and 1
    
    """
    
    bs, obj_num = eli_obj_idx.size()
    
    # Generate all time steps [0, time_steps] as candidate times
    all_times = torch.arange(1, max_action_time, step=interval, device=device).float()  # Do not include 0, as 0 has special meaning
    
    # Initialize disappear time, default is that all objects cannot disappear
    disappear_time = torch.full((bs, obj_num), max_game_time, device=device).float()
    
    # For each sample in the batch
    for i in range(bs):
        # setup_seed(100*trial_id+i)
        # Get the indices of objects in the current sample where eli_obj_idx is True
        mask = eli_obj_idx[i].bool()
        # Get the indices of objects that need to generate a time
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) > 0:
            # Shuffle all possible times
            shuffled_times = all_times[torch.randperm(all_times.size(0), device=device)]
            # Take the first len(valid_indices) times, ensuring they are unique
            chosen_times = shuffled_times[:len(valid_indices)]  # [len(valid_indices)]
            # Each object has epsilon probability of not disappearing
            disappear_mask = torch.rand(len(valid_indices), device=device) > epsilon  # [len(valid_indices)]
            chosen_times = chosen_times * disappear_mask.float() + max_game_time * (~disappear_mask).float()  # [len(valid_indices)]
            # Assign these times to the corresponding objects
            disappear_time[i, valid_indices] = chosen_times
    # Add an extra dimension when returning [bs, obj_num, 1]
    disappear_time = disappear_time.unsqueeze(-1)
    return disappear_time

class RefineDataset(torch.utils.data.Dataset):
    def __init__(self, body_property):
        self.body_property = body_property

        self.velocity = self.body_property[:,:,:,5:7] # [bs, steps, obj_num, 2]
        self.rotation_angle = self.body_property[:,:,:,7:8]
        self.angular_velocity = self.body_property[:,:,:,8:9]

        self.body_property = self.body_property[:,:,:,[0,1,2,3,4,9,10,11,12]] # [bs, steps, obj_num, 9]
        self.body_property = process_joint_length(self.body_property, obj_num)
        
    
    def __len__(self):
        return self.body_property.size(0)

    def __getitem__(self, idx):
        return self.body_property[idx], self.velocity[idx], self.rotation_angle[idx], self.angular_velocity[idx]



def refine(model,game_name, gt_simulation_trajectory, refine_epochs=201, trial_id=0):
    """
    gt_simulation_trajectory: [bs, steps, obj_num, 13维feature]
    disappear_time: [bs, obj_num, 1]
    """

    refine_dataset = RefineDataset(gt_simulation_trajectory)
    refine_loader = DataLoader(refine_dataset, batch_size=args.batch_size, shuffle=True)

    losses = []
    model.train()
    t1 = time.time()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = WeightedMSELoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(refine_epochs):
        epoch_loss,epoch_mse,epoch_res_loss,epoch_res_res_loss = 0,0,0,0
        SEGMENTS = args.seg
        for batch_idx, (body_property, velocity, rotation_angle, angular_velocity) in enumerate(refine_loader):
            body_property, velocity, rotation_angle, angular_velocity = body_property.to(device), velocity.to(device), rotation_angle.to(device), angular_velocity.to(device)
            bs, steps, obj_num = body_property.shape[0], body_property.shape[1], 12

            optimizer.zero_grad()

            t = torch.tensor([i/10 for i in range(time_steps)], device=device).float()  # 时间 [time_steps]

            assert steps % SEGMENTS == 0, "steps must be divisible by SEGMENTS"

            disappear_mask, dynamic_mask, done_mask, body_property, true_trajectories,angular_velocity = \
                                pre_process(body_property,rotation_angle,angular_velocity, bs, steps)

            body_property[:,:,:,3] /= args.angle_scale
            true_trajectories[:,:,:,3] /= args.angle_scale
            angular_velocity /= args.angle_scale

            stack_num = steps // SEGMENTS
            body_property_stacked, disappear_mask_stacked, velocity_stacked, angular_velocity_stacked, true_trajectories_stacked, done_mask_stacked = \
                                process_stacking_data(body_property,true_trajectories,disappear_mask,velocity,angular_velocity,done_mask,bs,steps,obj_num,SEGMENTS,stack_num)
            
            x0 = body_property_stacked[:, 0, :, :] 
            v0 = velocity_stacked[:, 0, :, :] 
            angular_velocity = angular_velocity_stacked[:, 0, :, :] 

            z0 = torch.cat([x0, v0,angular_velocity], dim=-1).to(device)

            t = t[:steps//SEGMENTS]
            t = t.repeat(bs * (SEGMENTS-1) * stack_num, 1).to(device)

            disappear_time_stacked = process_disappear_time(disappear_mask_stacked, steps, SEGMENTS)

            try:
                with torch.amp.autocast('cuda'):
                    predicted_trajectories = model(z0=z0, disappear_time=disappear_time_stacked, t=t)
            except:
                with torch.cuda.amp.autocast():
                    predicted_trajectories = model(z0=z0, disappear_time=disappear_time_stacked, t=t)
            
            predicted_trajectories = predicted_trajectories.reshape(stack_num*bs,steps//SEGMENTS*(SEGMENTS -1),obj_num,-1)
            true_trajectories_stacked = true_trajectories_stacked.reshape(stack_num*bs,steps//SEGMENTS*(SEGMENTS -1),obj_num,-1)
            done_mask_stacked = done_mask_stacked.reshape(stack_num*bs,steps//SEGMENTS*(SEGMENTS -1)).unsqueeze(-1).unsqueeze(-1)

            if epoch % 500 == 0 and epoch != 0:
                predicted_trajectories_for_draw = xyltheta_to_xyxy(predicted_trajectories, args.angle_scale)
                true_trajectories_for_draw = xyltheta_to_xyxy(true_trajectories_stacked, args.angle_scale)
                vis_trajectory(bs,obj_num,dynamic_mask,done_mask,predicted_trajectories_for_draw,true_trajectories_for_draw,save_name=f'{save_dir}/refine_trajectory_name={game_name}_batch={batch_idx}_epoch={epoch}',stride=steps//SEGMENTS)
            
            predicted_trajectories*= done_mask_stacked
            true_trajectories_stacked*= done_mask_stacked
            
            loss, mse, res_loss, res_res_loss = criterion(predicted_trajectories[:,:,:,:4], true_trajectories_stacked[:,:,:,:4])
            #loss.backward()
            scaler.scale(loss).backward()
            
            epoch_loss += loss
            epoch_mse += mse
            epoch_res_loss += res_loss
            epoch_res_res_loss += res_res_loss

        losses.append(epoch_loss.item())

        if epoch % 5 == 0:
            t2 = time.time()
            print(f"Refine Epoch {epoch}, Loss: {epoch_loss.item()}, MSE: {epoch_mse.item()}, Residual: {epoch_res_loss.item()}, Residual Residual: {epoch_res_res_loss.item()}, Time: {t2-t1}")
            logging.info(f"Refine Epoch {epoch}, Loss: {epoch_loss.item()}, MSE: {epoch_mse.item()}, Residual: {epoch_res_loss.item()}, Residual Residual: {epoch_res_res_loss.item()}, Time: {t2-t1}")
            t1 = t2
        
        scaler.step(optimizer)
        scaler.update()
    
    import matplotlib.pyplot as plt
    losses = np.array(losses)
    losses = np.log(losses)
    plt.plot(losses)
    plt.title(f"Refine Loss {game_name} (log scale)")
    plt.savefig(f'{save_dir}/refine_loss_{game_name}_trial_{trial_id}.png')
    plt.close()

    print(f"Refine {game_name} complete.")
    return model


test_set = IPHYREData_seq(data_path=f'../data/iphyre/game_seq_data', sample_ids=[0], game_ids=eval(args.games))
kwargs = {'pin_memory': True, 'num_workers': 0}
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

model.to(device)

# Testing loop
state_dict = torch.load(args.model_path)
if "module." in list(state_dict.keys())[0]:
    from collections import OrderedDict
    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())

model.load_state_dict(state_dict) 
model.eval()

all_results = {}


all_game_max_reward = {}
all_game_trial_reward = {}
for game_id, (game_names, _, body_property, actions, returns_to_go, timesteps, _,velocity,rotation_angle,angular_velocity) in enumerate(test_loader):

    model.load_state_dict(state_dict)
    model.eval()

    body_property,actions,returns_to_go,timesteps,velocity,rotation_angle,angular_velocity = body_property.to(device),actions.to(device),returns_to_go.to(device),timesteps.to(device),velocity.to(device),rotation_angle.to(device),angular_velocity.to(device)
    bs, steps, obj_num = body_property.shape[0], body_property.shape[1], 12  # [1, 150, 12, 9] 

    disappear_mask, dynamic_mask, done_mask, body_property, true_trajectories,angular_velocity = \
                        pre_process(body_property,rotation_angle,angular_velocity, bs, steps)

    body_property[:,:,:,3] /= args.angle_scale
    true_trajectories[:,:,:,3] /= args.angle_scale
    angular_velocity /= args.angle_scale
    
    x0 = body_property[:, 0, :, :] 
    v0 = velocity[:, 0, :, :] 
    angular_velocity = angular_velocity[:, 0, :, :] 

    # planning
    game_name = game_names[0]

    MAX_GUESSING = args.plan_sim_number
    max_reward = -999
    z0 = torch.cat([x0, v0,angular_velocity], dim=-1)  # [1, obj_num, feature_dim+3]

    t = torch.tensor([i/10 for i in range(time_steps)], device=device).float()  # [time_steps]
    t = t.unsqueeze(0).repeat(bs,1)

    z0 = z0.repeat(MAX_GUESSING, 1, 1) # [MAX_GUESSING, obj_num, FEATURE_DIM+3]
    t = t.repeat(MAX_GUESSING, 1) # [MAX_GUESSING,time_steps]

    eli_obj_idx = body_property[:,0,:,5].long().repeat(MAX_GUESSING, 1)   # [MAX_GUESSING, obj_num] 
    obj_num_mask = (body_property[:,0,:,:].sum(dim=-1) != 0).float().repeat(MAX_GUESSING, 1) # [MAX_GUESSING, obj_num]

    all_trial_observation = torch.tensor([])
    all_disappear_time = torch.tensor([]).to(device)

    # 5 trials
    for trial_idx in range(5):
        # mental simulation
        disappear_time = generate_disappear_time(eli_obj_idx, trial_id=trial_idx) # [MAX_GUESSING, obj_num, 1]
        disappear_time = disappear_time * obj_num_mask.unsqueeze(-1) # [MAX_GUESSING, obj_num, 1]

        with torch.no_grad():
            predicted_trajectories = model(z0, disappear_time, t) # [MAX_GUESSING, time_steps, obj_num, FEATURE_DIM+3+1]

        ball_mask = torch.where(predicted_trajectories[0, 0, :, 2].abs() < 1e-4, 1.0, 0.0) * torch.where(predicted_trajectories[0, 0, :, 4] > 0, 1.0, 0.0)
        is_ball_idx = torch.nonzero(ball_mask > 0.5).squeeze(-1)
        balls_min_height = predicted_trajectories[:,-1,is_ball_idx,1].min(-1).values # [MAX_GUESSING]
        top1_idx = balls_min_height.topk(1).indices
        
        # vis_gif(predicted_trajectories[top1_idx[0],...,:FEATURE_DIM].clone().reshape(-1,12,9), save_dir,save_jpg=False,file_name=f"mental_simulation_{game_name}_{trial_idx}.gif",normalized=True,forces=None,force_idx=None,draw_forces=False,angle_scale=args.angle_scale)

        # execute the top1 action sequence
        sim_fps = 60
        save_fps = 6
        env = IPHYRE(game_name, fps=sim_fps)
        env.reset()
        positions = env.get_action_space()
        positions = positions[1:]
        cur_dis_time = disappear_time[top1_idx[0],:,0]*10
        cur_dis_time *= eli_obj_idx[top1_idx[0]].float()
        cur_dis_time = cur_dis_time.detach().cpu().numpy().astype(int)
        cur_dis_time = cur_dis_time[(cur_dis_time != 0)]
        total_reward = 0
        true_trajectory = []
        max_time = 15
        sim_step = 0
        while sim_step < max_time * sim_fps:
            pos = [0., 0.]
            save_step = sim_step / save_fps
            if save_step in cur_dis_time:
                index = cur_dis_time.tolist().index(save_step)
                pos = positions[index]
            state, reward, done = env.step(pos)
            if sim_step % save_fps == 0:
                true_trajectory.append(state)
            total_reward += reward
            sim_step += 1
            if done:
                break
        max_reward = max(max_reward, total_reward)
        print(f"Game {game_name} trial {trial_idx} reward is {total_reward}")
        if game_name not in all_game_trial_reward.keys():
            all_game_trial_reward[game_name] = [total_reward]
        else:
            all_game_trial_reward[game_name].append(total_reward)
        
        true_trajectory = np.array(true_trajectory)
        true_trajectory[:,:,:5] /= 600
        true_trajectory[:,:,5:7] /= 600 # velocity

        true_trajectory = torch.tensor(true_trajectory).float().unsqueeze(0) # [1, time_steps, obj_num, FEATURE_DIM]
        true_trajectory = torch.cat([true_trajectory, torch.zeros(1, time_steps-true_trajectory.shape[1], obj_num, 13)], dim=1)

        true_trajectory_for_draw = xyxy_to_xyltheta(torch.cat([true_trajectory[...,:5],true_trajectory[...,9:]],dim=-1))
        # vis_gif(true_trajectory_for_draw[0,...,:FEATURE_DIM].clone().reshape(-1,12,9), save_dir,save_jpg=False,file_name=f"true_trajectory_{game_name}_{trial_idx}.gif",normalized=True,forces=None,force_idx=None,draw_forces=False,angle_scale=1) # 还没除，不需要angle_scale

        all_trial_observation = torch.cat([all_trial_observation, true_trajectory], dim=0)
        all_disappear_time = torch.cat([all_disappear_time, disappear_time[top1_idx[0]].unsqueeze(0)], dim=0)
        # refine the model based on the previous observed trajectories

        if trial_idx != 4: 
            model = refine(model,game_name, all_trial_observation, refine_epochs=args.num_epochs, trial_id=trial_idx)
            torch.save(model.state_dict(), f'{save_dir}/refined_model_{game_name}_{trial_idx}.pt')
            # evaluate the refined model
            refined_game_significance = evaluate_model(model,time_steps,f"[{PARAS_IDX[game_name]}]",args.dataset_name,top_number=args.eva_top_number,total_sim_number=args.eva_sim_number,evaluation_path=f'{save_dir}/evaluation_after_refine.csv',save_dir=save_dir,angle_scale=args.angle_scale)
            all_results[f'{game_id}_{game_name}_refine_after_trial_{trial_idx}'] = refined_game_significance    
    print(f"Game {game_name} max reward is {max_reward}")
    all_game_max_reward[game_name] = max_reward

print(all_game_max_reward)
print(f"Testing complete. Totally successed {(np.array(list(all_game_max_reward.values())) > 0).sum()} games. Mental simulation times is {args.plan_sim_number}.")
logging.info(f"Testing complete. Totally successed {(np.array(list(all_game_max_reward.values())) > 0).sum()} games. Mental simulation times is {args.plan_sim_number}.")

# save all_game_trial_reward
df = pd.DataFrame(all_game_trial_reward).T
df.to_csv(f'{save_dir}/all_game_trial_reward.csv')

for key in all_results.keys():
    # print(f"{key}: {all_results[key]}")
    df = pd.DataFrame(all_results[key]).T
    df.to_csv(f'{save_dir}/evaluation_result_{key}.csv')