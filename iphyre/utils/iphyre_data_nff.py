'''
Generate the offline dataset
'''

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import tqdm
import os
from iphyre.games import PARAS
from iphyre.simulator import IPHYRE
from utils.debug_draw import xyltheta_to_xyxy, xyxy_to_xyltheta


ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 2, 12 * 9, 256, 64, 64

class IPHYREData_seq(Dataset):
    def __init__(self, data_path, sample_ids, game_ids, max_len=150):
        self.data_path = data_path
        self.sample_ids = sample_ids
        self.max_len = max_len
        self.num_games = len(PARAS)

        self.split = list(PARAS.keys())
        self.split = [self.split[i] for i in game_ids]
        self.game_names = []
        self.sequence_scenes = []
        self.body_property = []
        self.velocity = []
        self.rotation_angle = []
        self.angular_velocity = []
        # self.eli_body_property = []
        self.actions = []
        self.returns_to_go = []
        self.time_steps = []
        self.body_num = []
        self.done = []

        self.target_return = 1

        for game in self.split:
            print(f'loading {game}')
            if not os.path.exists(f'{data_path}/{game}'):
                continue
            env = IPHYRE(game, fps=10)
            self.game_names += [game] * len(self.sample_ids)
            for idx in self.sample_ids:
                path = f'{data_path}/{game}/{idx}'
                property_path = path + '/' + 'vectors.npy'
                # eli_property_path = path + '/' + 'eli_test.npy'
                actions_path = path + '/' + 'actions.npy'
                body_property = np.load(property_path)

                actions = np.load(actions_path)
                # eli_property = np.load(eli_property_path)
                positions = env.get_action_space()

                assert self.max_len == body_property.shape[0]
                # convert actions (x, y, t) to [...,[x, y],....]
                actions_seq = np.zeros((self.max_len, 2))
                for a in actions:
                    actions_seq[int(a[-1] * 10)] = a[0:2]

                # convert actions_seq to one hot
                one_hot_actions_seq = np.zeros((self.max_len, 7))
                for i, a in enumerate(actions_seq):
                    id = positions.index(a.tolist())
                    one_hot_actions_seq[i][id] = 1.

                # velocity
                velocity = body_property[:,:,5:7].copy()
                velocity /= 600

                # rotation_angle
                rotation_angle = body_property[:,:,7:8].copy()

                # angular velocity
                angular_velocity = body_property[:,:,8:9].copy()

                # normalize
                body_property = np.concatenate((body_property[:,:,:5],body_property[:,:,9:]),-1)
                body_property[:,:,:5] /= 600

                obj_num = body_property.shape[1]
                joint_idxs = body_property[:, :, 7]
                nonzero_mask = joint_idxs != 0
                idxs_expanded_1 = joint_idxs[:, :, np.newaxis]
                idxs_expanded_2 = joint_idxs[:, np.newaxis, :]
                equality_mask = idxs_expanded_1 == idxs_expanded_2
                nonzero_mask_expanded_1 = nonzero_mask[:, :, np.newaxis]
                nonzero_mask_expanded_2 = nonzero_mask[:, np.newaxis, :]
                combined_mask = equality_mask & nonzero_mask_expanded_1 & nonzero_mask_expanded_2
                triu_mask = np.triu(np.ones((obj_num, obj_num), dtype=bool), k=1)
                pair_mask = combined_mask & triu_mask[np.newaxis, :, :]
                time_indices, obj_idx1, obj_idx2 = np.nonzero(pair_mask)
                features1 = body_property[time_indices, obj_idx1, :]
                features2 = body_property[time_indices, obj_idx2, :]
                centroids1 = (features1[:, 0:2] + features1[:, 2:4]) / 2
                centroids2 = (features2[:, 0:2] + features2[:, 2:4]) / 2
                distances = np.linalg.norm(centroids1 - centroids2, axis=1)
                idx_values = features1[:, 7]
                new_idxs = idx_values + distances
                body_property[time_indices, obj_idx1, 7] = new_idxs
                body_property[time_indices, obj_idx2, 7] = new_idxs

                seq_len = 0
                for seq in body_property:
                    if seq.sum() != 0:
                        seq_len += 1

                self.velocity.append(velocity)
                self.rotation_angle.append(rotation_angle)
                self.angular_velocity.append(angular_velocity)

                self.sequence_scenes.append(0)
                self.body_property.append(body_property)
                # self.eli_body_property.append(eli_property)
                self.actions.append(one_hot_actions_seq)
                if idx < 50:
                    self.returns_to_go.append(
                        [[self.target_return] for i in range(seq_len)] + [[0] for _ in range(self.max_len - seq_len)])
                    self.done.append(
                        [[0] for _ in range(seq_len)] + [[1]] + [[1] for _ in range(self.max_len - seq_len - 1)])
                else:
                    self.returns_to_go.append([[self.target_return] for _ in range(self.max_len)])
                    self.done.append([[0] for _ in range(self.max_len)])

                self.time_steps.append([[t*15 / self.max_len] for t in range(self.max_len)])  # norm time steps
                self.body_num.append(0)


        self.body_property = np.array(self.body_property, dtype=np.float32)
        self.sequence_scenes = self.body_property
        self.actions = np.array(self.actions, dtype=np.float32)
        self.returns_to_go = np.array(self.returns_to_go, dtype=np.float32)
        self.time_steps = np.array(self.time_steps, dtype=np.float32)
        self.body_num = np.array(self.body_num, dtype=np.float32)
        self.velocity = np.array(self.velocity, dtype=np.float32)
        self.rotation_angle = np.array(self.rotation_angle, dtype=np.float32)
        self.angular_velocity = np.array(self.angular_velocity, dtype=np.float32)


    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return self.game_names[idx], \
               self.sequence_scenes[idx], \
               self.body_property[idx], \
               self.actions[idx], \
               self.returns_to_go[idx], \
               self.time_steps[idx], \
               self.body_num[idx], \
               self.velocity[idx], \
               self.rotation_angle[idx], \
               self.angular_velocity[idx]

def pre_process(body_property,rotation_angle,angular_velocity, bs, steps):
    # extract disappear_mask and dynamic_mask and done_mask
    disappear_mask = (body_property[:, :, :, :].sum(dim=-1) != 0)
    dynamic_mask = body_property[:, :, :, -3:-2]
    done_mask = (body_property.reshape(bs,steps,-1).sum(dim=-1) != 0)
    done_mask = done_mask.float()
    
    # transform properties
    body_property = xyxy_to_xyltheta(body_property)
    not_ball_mask = (body_property[:,:,:,2:3].abs() > 1e-5).float()
    angular_velocity = angular_velocity * not_ball_mask
    rotation_angle = rotation_angle * not_ball_mask # mask ball rotation

    body_property[:,:,:,3:4] = body_property[:,0:1,:,3:4] + rotation_angle
    body_property *= disappear_mask.unsqueeze(-1).float()

    true_trajectories = body_property[:, :, :, :].clone()

    return disappear_mask, dynamic_mask, done_mask, body_property, true_trajectories,angular_velocity

def process_joint_length(body_property, obj_num):
    joint_idxs = body_property[:,:,:,7]# [1, 150, 12]
    nonzero_mask = joint_idxs != 0 
    idxs_expanded_1 = joint_idxs.unsqueeze(3)  # [batch_size, time_steps, obj_num, 1]
    idxs_expanded_2 = joint_idxs.unsqueeze(2)  # [batch_size, time_steps, 1, obj_num]

    equality_mask = (idxs_expanded_1 == idxs_expanded_2)  # [batch_size, time_steps, obj_num, obj_num]
    nonzero_mask_expanded_1 = nonzero_mask.unsqueeze(3)  # [batch_size, time_steps, obj_num, 1]
    nonzero_mask_expanded_2 = nonzero_mask.unsqueeze(2)  # [batch_size, time_steps, 1, obj_num]
    combined_mask = equality_mask & nonzero_mask_expanded_1 & nonzero_mask_expanded_2  # [batch_size, time_steps, obj_num, obj_num]

    triu_mask = torch.triu(torch.ones(obj_num, obj_num, dtype=torch.bool), diagonal=1).to(body_property.device)  # [obj_num, obj_num]
    triu_mask = triu_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, obj_num, obj_num]

    pair_mask = combined_mask & triu_mask  # [batch_size, time_steps, obj_num, obj_num]
    batch_indices, time_indices, obj_idx1, obj_idx2 = torch.nonzero(pair_mask, as_tuple=True)

    features1 = body_property[batch_indices, time_indices, obj_idx1, :]  # [num_pairs, feature_dim]
    features2 = body_property[batch_indices, time_indices, obj_idx2, :]  # [num_pairs, feature_dim]

    centroids1 = (features1[:, 0:2] + features1[:, 2:4]) / 2  # [num_pairs, 2]
    centroids2 = (features2[:, 0:2] + features2[:, 2:4]) / 2  # [num_pairs, 2]

    distances = torch.norm(centroids1 - centroids2, dim=1)  # [num_pairs]
    idx_values = features1[:, 7]
    new_idxs = idx_values + distances

    body_property[batch_indices, time_indices, obj_idx1, 7] = new_idxs
    body_property[batch_indices, time_indices, obj_idx2, 7] = new_idxs

    return body_property



def process_disappear_time(disappear_mask, steps, SEGMENTS):
    disappear_indices = (~disappear_mask).int()
    disappear_time = torch.argmax(disappear_indices, dim=1, keepdim=False).float()
    # If an object never disappears, set disappear_time to the maximum time step
    never_disappear = disappear_mask.all(dim=1, keepdim=False)
    disappear_time = torch.where(never_disappear, torch.tensor(steps//SEGMENTS).float().to(never_disappear.device), disappear_time)
    disappear_time = disappear_time.unsqueeze(-1)  # Shape: [bs*stack_num*(SEGMENTS-1), obj_num, 1]
    disappear_time /= 10.0

    return disappear_time



def process_stacking_data(body_property, true_trajectories, disappear_mask, velocity, angular_velocity, done_mask, bs, steps, obj_num, SEGMENTS,stack_num):
    indices = torch.arange(0, steps - steps//SEGMENTS).unsqueeze(0) + torch.arange(0, steps//SEGMENTS).unsqueeze(1)
    body_property_stacked = body_property[:, indices, :, :]
    disappear_mask_stacked = disappear_mask[:, indices, :]
    velocity_stacked = velocity[:, indices, :, :]
    angular_velocity_stacked = angular_velocity[:, indices, :, :]

    true_trajectories_stacked = true_trajectories[:, indices, :, :]
    done_mask_stacked = done_mask[:, indices]

    body_property_stacked = body_property_stacked.reshape(bs*stack_num, (SEGMENTS-1), steps//SEGMENTS, obj_num, 9) # [1, 10, 15, 12, 9]
    body_property_stacked = body_property_stacked.reshape(bs*stack_num*(SEGMENTS-1), -1 , obj_num, 9) # [10, 15, 12, 9]

    disappear_mask_stacked = disappear_mask_stacked.reshape(bs*stack_num,(SEGMENTS-1), steps//SEGMENTS, obj_num) # [1, 10, 15, 12]
    disappear_mask_stacked = disappear_mask_stacked.reshape(bs*stack_num*(SEGMENTS-1), -1, obj_num) # [10, 15, 12]

    velocity_stacked = velocity_stacked.reshape(bs*stack_num, (SEGMENTS-1), steps//SEGMENTS, obj_num, 2) # [1, 10, 15, 12, 2]
    velocity_stacked = velocity_stacked.reshape(bs*stack_num*(SEGMENTS-1), -1, obj_num, 2) # [10, 15, 12, 2]

    angular_velocity_stacked = angular_velocity_stacked.reshape(bs*stack_num, (SEGMENTS-1), steps//SEGMENTS, obj_num, 1) # [1, 10, 15, 12, 1]
    angular_velocity_stacked = angular_velocity_stacked.reshape(bs*stack_num*(SEGMENTS-1), -1, obj_num, 1) # [10, 15, 12, 1]

    return body_property_stacked, disappear_mask_stacked, velocity_stacked, angular_velocity_stacked, true_trajectories_stacked, done_mask_stacked