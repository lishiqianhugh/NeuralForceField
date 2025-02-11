import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import os
import torch
from PIL import Image
from utils.util import save_tensor_as_csv
from iphyre.games import PARAS
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
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

def draw_arrow(surface, start_pos, end_pos, color, arrow_size=10):
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    angle = math.atan2(dy, dx)
    length = math.hypot(dx, dy)

    end_pos = (float(end_pos[0]), float(end_pos[1]))
    start_pos = (float(start_pos[0]), float(start_pos[1]))

    pygame.draw.line(surface, color, start_pos, end_pos, 3)

    arrow_point1 = (end_pos[0] - arrow_size * math.cos(angle - math.pi / 6),
                   end_pos[1] - arrow_size * math.sin(angle - math.pi / 6))
    arrow_point2 = (end_pos[0] - arrow_size * math.cos(angle + math.pi / 6),
                   end_pos[1] - arrow_size * math.sin(angle + math.pi / 6))

    pygame.draw.polygon(surface, color, [end_pos, arrow_point1, arrow_point2])

def render_image_from_vectors(vectors, width=600, height=600,normalized=True,draw_forces = False,forces = None,force_idx = None):
    """
    vectors : [obj_num,9]
    forces : [force_idx,force_obj_num,2] 受力物体数，施力物体数
    """
    if draw_forces and (forces is None or force_idx is None):
        raise ValueError("You must provide forces and force_idx if draw_forces is True")
    
    if normalized:
        vectors[:,:3] *= 600
        vectors[:,4:5] *= 600
        if forces is not None:
            forces *= 2400

    # Initialize Pygame and pymunk
    pygame.init()
    screen = pygame.Surface((width, height))

    # Create a pymunk space
    space = pymunk.Space()
    space.gravity = (0, 100)  # Set gravity as in the original code
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Fill the screen with white
    screen.fill((255, 255, 255))

    bodies = []  # Store bodies for later use in joints and springs

    for i, obj in enumerate(vectors):
        center_x, center_y, length, angle, r, eli, dynamic, joint, spring = obj
        # print(center_x, center_y, length, angle, r, eli, dynamic, joint, spring)
        if length == 0:  # It's a ball
            if dynamic:
                body = pymunk.Body(mass=1.0, moment=pymunk.moment_for_circle(1.0, 0, r))
            else:
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = center_x, center_y
            body.angle = angle
            shape = pymunk.Circle(body, r)
            shape.color = (255, 0, 0, 255)
            shape.elasticity = 0.1
            shape.friction = 0.5
        else:  # It's a block (line segment)
            if dynamic:
                mass = 1.0
                moment = pymunk.moment_for_segment(mass, ( -length / 2, 0), (length / 2, 0), 10)
                body = pymunk.Body(mass=mass, moment=moment)
            else:
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
            
            body.position = center_x, center_y
            body.angle = angle
            shape = pymunk.Segment(body, (-length / 2, 0), (length / 2, 0), 10)
            if eli:
                shape.color = (164, 164, 164, 255)
            elif dynamic:
                shape.color = (90, 148, 220, 255)
            else:
                shape.color = (0, 0, 0, 255)
            shape.elasticity = 0.1
            shape.friction = 0.5

        space.add(body, shape)
        bodies.append(body)

    # Add joints and springs if specified
    for i, obj in enumerate(vectors):
        _, _, _, _, _, _, _, joint, spring = obj
        if joint:
            for j in range(i+1, len(vectors)):
                if vectors[j][7] == joint:
                    c = pymunk.PinJoint(bodies[i], bodies[j])
                    space.add(c)
                    break
        if spring:
            for j in range(i+1, len(vectors)):
                if vectors[j][8] == spring:
                    c = pymunk.DampedSpring(bodies[i], bodies[j], (0, 0), (0, 0), 20, 1, 0.3)
                    space.add(c)
                    break

    # Draw the space
    space.debug_draw(draw_options)

    # Draw forces using quiver
    if draw_forces:
        for i, idx in enumerate(force_idx):
            for force in forces[i]:
                if force.sum() == 0:
                    continue
                draw_arrow(screen, (bodies[idx].position.x, bodies[idx].position.y), (bodies[idx].position.x + force[0], bodies[idx].position.y + force[1]), (255, 165, 0), arrow_size=10)

    return pygame.surfarray.array3d(screen)

def vis_gif(vectors, output_folder, forces, force_idx, save_jpg=False,file_name="animation.gif",normalized=True,width=600, height=600, draw_forces=True,angle_scale=1e2):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    vectors = vectors.cpu().detach().numpy()
    vectors[...,3] *= angle_scale
    if draw_forces:
        forces = forces.cpu().detach().numpy()
    # List to hold frames for GIF
    gif_frames = []

    # Render and save each frame
    for i, frame_vectors in enumerate(vectors):
        if frame_vectors.sum() == 0:
            break
        
        if draw_forces:
            frame_array = render_image_from_vectors(frame_vectors,normalized=normalized,width=width, height=height,draw_forces = draw_forces,forces = forces[i],force_idx = force_idx)
        else:
            frame_array = render_image_from_vectors(frame_vectors,normalized=normalized,width=width, height=height,draw_forces = draw_forces,forces=None,force_idx=None)
        
        # Convert to correct color format and orientation
        frame = pygame.surfarray.make_surface(frame_array)
        
        # Save the image as JPG
        if save_jpg:
            image_path = os.path.join(output_folder, f"frame_{i:04d}.jpg")
            pygame.image.save(frame, image_path)
            print(f"Saved frame {i}")

        # Convert the frame for the GIF and add it to the list
        gif_frame = Image.fromarray(np.transpose(frame_array, (1, 0, 2)))
        gif_frames.append(gif_frame)

    # Save the frames as a GIF
    gif_path = os.path.join(output_folder, file_name)
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)

    pygame.quit()

from matplotlib.patches import Rectangle, Circle
def add_capsule(ax, point1, point2, radius, edgecolor="black", linewidth=2):

    if point1 == point2:
        color = (1, 182/255, 193/255, 1)
    else:
        color = (90/255, 148/255, 220/255, 1)

    vector = np.array(point2) - np.array(point1)
    length = np.linalg.norm(vector)
    direction = vector / length

    perp_direction = np.array([-direction[1], direction[0]])

    angle = np.degrees(np.arctan2(direction[1], direction[0]))

    rect_start = np.array(point1) - radius * perp_direction

    rect_width = length
    rect_height = 2 * radius

    rect = Rectangle(
        rect_start,
        rect_width,
        rect_height,
        angle=angle,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=0
    )
    ax.add_patch(rect)

    left_circle_center = np.array(point1)
    left_circle = Circle(
        left_circle_center,
        radius,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=0
    )
    ax.add_patch(left_circle)

    right_circle_center = np.array(point2)
    right_circle = Circle(
        right_circle_center,
        radius,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=0
    )
    ax.add_patch(right_circle)



def xyltheta_to_xyxy(trajectories, angle_scale):
    assert len(trajectories.shape) == 4, "Input shape must be [batch_size, time_steps, obj_num, >=4]"
    assert trajectories.shape[-1] >= 4, "Input shape must be [batch_size, time_steps, obj_num, >=4]"

    central_point = trajectories[:,:,:,:2].clone()
    block_length = trajectories[:,:,:,2:3].clone()
    rot_angle = trajectories[:,:,:,3:4].clone() * angle_scale

    transformed_trajectories = trajectories.clone()
    transformed_trajectories[:,:,:,:2] = central_point - block_length/2 * torch.cat([torch.cos(rot_angle),torch.sin(rot_angle)],dim=-1)
    transformed_trajectories[:,:,:,2:4] = central_point + block_length/2 * torch.cat([torch.cos(rot_angle),torch.sin(rot_angle)],dim=-1)

    return transformed_trajectories

def xyxy_to_xyltheta(trajectories):
    assert len(trajectories.shape) == 4, "Input shape must be [batch_size, time_steps, obj_num, >=4]"
    assert trajectories.shape[-1] >= 4, "Input shape must be [batch_size, time_steps, obj_num, >=4]"

    central_point = (trajectories[:,:,:,:2].clone() + trajectories[:,:,:,2:4].clone()) / 2
    block_length = torch.norm(trajectories[:,:,:,:2].clone() - trajectories[:,:,:,2:4].clone(),dim=-1,keepdim=True)
    rot_angle = torch.atan2(trajectories[:,:,:,3:4].clone()-trajectories[:,:,:,1:2].clone(),trajectories[:,:,:,2:3].clone()-trajectories[:,:,:,0:1].clone())

    transformed_trajectories = trajectories.clone()
    transformed_trajectories[:,:,:,:2] = central_point
    transformed_trajectories[:,:,:,2:3] = block_length
    transformed_trajectories[:,:,:,3:4] = rot_angle

    return transformed_trajectories


def vis_trajectory(batch_size,obj_num,dynamic_mask,done_mask,predicted_trajectories_for_draw,true_trajectories_for_draw,save_name,stride=1):
    done_time = done_mask.sum(dim=1).int()
    for batchidx in range(batch_size):
        seaborn_colors = sns.color_palette("bright", 12)
        colors = np.array([list(color) + [1.0] for color in seaborn_colors])
        fig = plt.figure(figsize=(16, 16))
        # for each object
        for i in range(obj_num):
            # if dynamic
            color = colors[i]
            #color_gt = colors_gt[i]
            if dynamic_mask[batchidx,0,i,0] == 1:
                plt.scatter(predicted_trajectories_for_draw[batchidx*(stride), :done_time[batchidx], i, 0].detach().cpu().numpy(), 
                            1 - predicted_trajectories_for_draw[batchidx*(stride), :done_time[batchidx], i, 1].detach().cpu().numpy(), 
                            label=f'Predicted Object {i+1}', 
                            marker='x', color=color,zorder = 10)
                plt.scatter(predicted_trajectories_for_draw[batchidx*(stride), :done_time[batchidx], i, 2].detach().cpu().numpy(), 
                            1 - predicted_trajectories_for_draw[batchidx*(stride), :done_time[batchidx], i, 3].detach().cpu().numpy(), 
                            label=f'Predicted Object {i+1}', 
                            marker='x', color=color,zorder = 10)
                plt.scatter(true_trajectories_for_draw[batchidx*(stride),:done_time[batchidx],i,0].detach().cpu().numpy(), 
                            1- true_trajectories_for_draw[batchidx*(stride),:done_time[batchidx],i,1].detach().cpu().numpy(),
                            label=f'GT Object {i+1}', marker='o', color="black",zorder = 1)
                plt.scatter(true_trajectories_for_draw[batchidx*(stride),:done_time[batchidx],i,2].detach().cpu().numpy(), 
                            1- true_trajectories_for_draw[batchidx*(stride),:done_time[batchidx],i,3].detach().cpu().numpy(),
                            label=f'GT Object {i+1}', marker='o', color="black",zorder = 1)
            
            ax = fig.gca()
            add_capsule(ax, (true_trajectories_for_draw[batchidx*(stride),0,i,0].detach().cpu().numpy(), 1-true_trajectories_for_draw[batchidx*(stride),0,i,1].detach().cpu().numpy()), 
                        (true_trajectories_for_draw[batchidx*(stride),0,i,2].detach().cpu().numpy(), 1-true_trajectories_for_draw[batchidx*(stride),0,i,3].detach().cpu().numpy()), 
                        true_trajectories_for_draw[batchidx*(stride),0,i,4].detach().cpu().numpy(),edgecolor=colors[i],linewidth=5)
        # plt.legend()
        plt.xlim(0,1)
        plt.ylim(0,1) # set xy lim
        plt.savefig(save_name+f'_mini_batch={batchidx}.png')
        plt.close()

def vis_forces(batch_size,obj_num,dynamic_mask,done_mask,predicted_trajectories_for_draw,true_trajectories_for_draw,predicted_trajectories,predicted_velocities,force_predictor,gravity_value,save_name,stride=1):
    done_time = done_mask.sum(dim=1).int()
    for batchidx in range(batch_size):
        seaborn_colors = sns.color_palette("bright", 12)
        colors = np.array([list(color) + [1.0] for color in seaborn_colors])
        for i in range(obj_num):
            # if dynamic
            if dynamic_mask[batchidx,0,i] != 1:
                continue
            fig = plt.figure(figsize=(16, 16))
            if dynamic_mask[batchidx,0,i] == 1:
                x_coords_1 = predicted_trajectories_for_draw[batchidx * stride, :done_time[batchidx], i, 0].detach().cpu().numpy()
                y_coords_1 = 1 - predicted_trajectories_for_draw[batchidx * stride, :done_time[batchidx], i, 1].detach().cpu().numpy()
                x_coords_2 = predicted_trajectories_for_draw[batchidx * stride, :done_time[batchidx], i, 2].detach().cpu().numpy()
                y_coords_2 = 1 - predicted_trajectories_for_draw[batchidx * stride, :done_time[batchidx], i, 3].detach().cpu().numpy()
                plt.scatter(x_coords_1, y_coords_1, label=f'Predicted Object {i+1}', marker='x', color=colors[i], zorder=10)
                plt.scatter(x_coords_2, y_coords_2, marker='x', color=colors[i], zorder=10)

                predicted_force,_ = force_predictor(init_x=predicted_trajectories[batchidx*(stride),:done_time[batchidx],:,:9].reshape(-1,12,9),
                                                query_x=predicted_trajectories[batchidx*(stride),:done_time[batchidx],i,:9].reshape(-1,1,9),
                                                init_v=predicted_velocities[batchidx*(stride),:done_time[batchidx],:,:2].reshape(-1,12,2),
                                                query_v=predicted_velocities[batchidx*(stride),:done_time[batchidx],i,:2].reshape(-1,1,2),
                                                init_angular_v=predicted_velocities[batchidx*(stride),:done_time[batchidx],:,-1].reshape(-1,12,1),
                                                query_angular_v=predicted_velocities[batchidx*(stride),:done_time[batchidx],i,2].reshape(-1,1,1))
                for k in range(obj_num):
                    if k != i:
                        color = colors[k]
                        x = predicted_trajectories[batchidx*(stride), :done_time[batchidx], i, 0].detach().cpu().numpy() # [1, 150, 12, 9]
                        y = 1 - predicted_trajectories[batchidx*(stride), :done_time[batchidx], i, 1].detach().cpu().numpy()
                        plt.quiver(x,y,
                                    predicted_force[:,k,:,0].detach().cpu().numpy(), # [150, 12, 1, 3]
                                    -predicted_force[:,k,:,1].detach().cpu().numpy(),
                                    pivot='tail',width=0.002,scale=0.3,color = color,
                                    headwidth=6,        # Increase head width
                                    headaxislength=6,   # Increase the length of the head axis
                                    headlength=6       # Adjust head length (optional)
                                    )
                        torque_magnitude = abs(predicted_force[:,k,:,2].detach().cpu().numpy())
                        torque_magnitude = torque_magnitude / (torque_magnitude.max()+1e-6) * 0.005
                        for j in range(x.shape[0]):
                            if torque_magnitude[j] < 0:
                                circle = patches.Circle(
                                    (x[j], y[j]),                # Center of the circle
                                    radius=torque_magnitude[j],  # Radius based on torque magnitude
                                    facecolor='red',
                                    linewidth=0,             # Line width of the circle edge
                                    zorder = 3
                                )
                            else:
                                circle = patches.Circle(
                                    (x[j], y[j]),                # Center of the circle
                                    radius=torque_magnitude[j],  # Radius based on torque magnitude
                                    facecolor='blue',
                                    linewidth=0,             # Line width of the circle edge
                                    zorder = 3
                                )
                            # Add the circle to the current axes
                            plt.gca().add_patch(circle)
                    
                plt.quiver(predicted_trajectories[batchidx*(stride), :done_time[batchidx], i, 0].detach().cpu().numpy(),
                            1 - predicted_trajectories[batchidx*(stride), :done_time[batchidx], i, 1].detach().cpu().numpy(),
                            0,
                            gravity_value,
                            pivot='tail', color='black',width=0.002,scale=0.3,
                            headwidth=6,        
                            headaxislength=6,
                            headlength=6      
                            )
                    
                    
            if dynamic_mask[batchidx,0,i] == 1:
                plt.scatter(true_trajectories_for_draw[batchidx*(stride),:done_time[batchidx],i,0].detach().cpu().numpy(), 
                                1- true_trajectories_for_draw[batchidx*(stride),:done_time[batchidx],i,1].detach().cpu().numpy(),
                                label=f'GT Object {i+1}', marker='o',color="black",zorder = 1)
                plt.scatter(true_trajectories_for_draw[batchidx*(stride),:done_time[batchidx],i,2].detach().cpu().numpy(), 
                            1- true_trajectories_for_draw[batchidx*(stride),:done_time[batchidx],i,3].detach().cpu().numpy(),
                            label=f'GT Object {i+1}', marker='o',color="black",zorder = 1)
            ax = fig.gca()
            for t in range(obj_num):
                add_capsule(ax, (true_trajectories_for_draw[batchidx*(stride),0,t,0].detach().cpu().numpy(), 1-true_trajectories_for_draw[batchidx*(stride),0,t,1].detach().cpu().numpy()), 
                            (true_trajectories_for_draw[batchidx*(stride),0,t,2].detach().cpu().numpy(), 1-true_trajectories_for_draw[batchidx*(stride),0,t,3].detach().cpu().numpy()), 
                            true_trajectories_for_draw[batchidx*(stride),0,t,4].detach().cpu().numpy(),edgecolor=colors[t],linewidth=5)
            # plt.legend()
            # set xy lim
            plt.xlim(0,1)
            plt.ylim(0,1)
            # no sticks
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_name+f'_mini_batch={batchidx}_object={i}.png', bbox_inches='tight', dpi=600)
            plt.close()


def vis_force_field(predicted_trajectories, predicted_velocities, force_predictor, save_dir, steps, device, obj_num, batchidx):
    resolution = 200
    # query_x = sample resolution*resolution grid in the range of [0,1]x[0,1]
    init_x = predicted_trajectories[batchidx,:,:,:].reshape(-1,obj_num,9).to(device)
    query_x = torch.zeros(resolution*resolution,9).to(device)
    query_x[:,0] = torch.linspace(0,1,resolution).unsqueeze(1).repeat(1,resolution).reshape(-1)
    query_x[:,1] = torch.linspace(0,1,resolution).repeat(resolution).reshape(-1)
    query_x[:,4] = 0.033 # ball radius
    query_x = query_x.unsqueeze(0).repeat(steps,1,1)
    # query_v = zeros query_angular_v = zeros
    init_v = predicted_velocities[batchidx,:,:,:2].reshape(-1,obj_num,2)
    query_v = torch.zeros(steps,resolution*resolution,2).to(device)
    # query_v[...,0] += 1
    init_angular_v = predicted_velocities[batchidx,:,:,-1].reshape(-1,obj_num,1)
    query_angular_v = torch.zeros(steps,resolution*resolution,1).to(device)
    with torch.no_grad():
        predicted_force,_ = force_predictor(init_x=init_x,
                                        query_x=query_x,
                                        init_v=init_v,
                                        query_v=query_v,
                                        init_angular_v=init_angular_v,
                                        query_angular_v=query_angular_v)
    

    predicted_force *= 20
    print(f'Predicted Force: max = {predicted_force[:,:,0,:2].max()} min = {predicted_force[:,:,0,:2].min()}')
    print(f'Predicted torque: max = {predicted_force[:,:,0,2].max()} min = {predicted_force[:,:,0,2].min()}')
    not_ball_mask = (init_x[...,2:3].abs() > 1e-5).float().unsqueeze(-2).repeat(1,1,resolution*resolution,1)
    predicted_force = predicted_force * not_ball_mask
    predicted_force = predicted_force.reshape(steps,obj_num,resolution,resolution,3).sum(dim=1)
    for t in range(2):
        visualize_field(predicted_force[t].cpu().detach().numpy(), file_name=f'{save_dir}/force_field_batch={batchidx}_step={t}.png', rot=True)

def lic_flow(vectors_x, vectors_y, texture, magnitudes, kernel_length=10):
    h, w = texture.shape
    result = np.zeros_like(texture)
    magnitude_result = np.zeros_like(texture)
    
    for i in range(h):
        for j in range(w):
            x, y = j, i
            forward_sum = texture[i,j]
            backward_sum = texture[i,j]
            forward_mag_sum = magnitudes[i,j]
            backward_mag_sum = magnitudes[i,j]
            weight_sum = 1.0
            
            # Forward integration
            for k in range(kernel_length):
                vx = vectors_x[int(y), int(x)]
                vy = vectors_y[int(y), int(x)]
                v_mag = np.sqrt(vx*vx + vy*vy)
                if v_mag == 0: break
                
                vx, vy = vx/v_mag, vy/v_mag
                x += vx
                y += vy
                
                if x < 0 or x >= w-1 or y < 0 or y >= h-1: break
                forward_sum += texture[int(y), int(x)]
                forward_mag_sum += magnitudes[int(y), int(x)]
                weight_sum += 1
            
            x, y = j, i
            # Backward integration
            for k in range(kernel_length):
                vx = vectors_x[int(y), int(x)]
                vy = vectors_y[int(y), int(x)]
                v_mag = np.sqrt(vx*vx + vy*vy)
                if v_mag == 0: break
                
                vx, vy = -vx/v_mag, -vy/v_mag
                x += vx
                y += vy
                
                if x < 0 or x >= w-1 or y < 0 or y >= h-1: break
                backward_sum += texture[int(y), int(x)]
                backward_mag_sum += magnitudes[int(y), int(x)]
                weight_sum += 1
            
            result[i,j] = (forward_sum + backward_sum) / weight_sum
            magnitude_result[i,j] = (forward_mag_sum + backward_mag_sum) / weight_sum
    
    return result, magnitude_result


def visualize_field(vector_field, extent=[-2, 2, -2, 2], figsize=(10, 10), file_name='field.png', rot=False):
    """
    Visualize a vector field using Line Integral Convolution (LIC) with magnitude representation
    
    Args:
        vector_field: numpy array of shape [H, W, 2] containing vector components
        extent: plot extent [xmin, xmax, ymin, ymax]
        figsize: figure size tuple
        file_name: output file name
        rot: whether to rotate the field
    """
    h, w = vector_field.shape[:2]
    
    vectors_x = vector_field[..., 0]
    vectors_y = vector_field[..., 1]
    
    if rot:
        vectors_x = np.rot90(vectors_x)
        vectors_y = np.rot90(vectors_y)
    
    # Calculate magnitude before normalization
    magnitude = np.sqrt(vectors_x**2 + vectors_y**2)
    
    # Normalize vectors for direction
    vectors_x_norm = np.divide(vectors_x, magnitude, where=magnitude != 0)
    vectors_y_norm = np.divide(vectors_y, magnitude, where=magnitude != 0)
    
    # random texture
    np.random.seed(42)
    texture = np.random.rand(h, w)
    
    lic_result, mag_result = lic_flow(vectors_x_norm, vectors_y_norm, texture, magnitude)
    
    mag_normalized = np.log1p(mag_result)  # Using log scale for better visualization
    mag_normalized = (mag_normalized - np.min(mag_normalized)) / (np.max(mag_normalized) - np.min(mag_normalized))
    
    # Combine LIC and magnitude
    combined_result = lic_result * mag_normalized
    
    plt.figure(figsize=figsize)
    plt.imshow(combined_result, cmap='inferno', extent=extent)
    plt.gca().invert_yaxis()
    # set colobar text size
    cbar = plt.colorbar(label='Magnitude', fraction=0.046, pad=0.04, orientation='horizontal')
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.set_xlabel('Magnitude', fontsize=40)
    # plt.title('Field Visualization (Direction + Magnitude)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.axis('equal')
    # remove border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # no ticks
    plt.xticks([])
    plt.yticks([])
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    return

def calculate_electric_field(x, y, charges):
    k = 8.99e9
    Ex, Ey = np.zeros_like(x), np.zeros_like(y)
    
    for charge in charges:
        dx = x - charge['pos'][0]
        dy = y - charge['pos'][1]
        r = np.sqrt(dx**2 + dy**2)
        r = np.where(r == 0, 1e-12, r)
        magnitude = k * charge['q'] / (r**2)
        Ex += magnitude * dx/r
        Ey += magnitude * dy/r
    
    return np.stack([Ex, Ey], axis=-1)

if __name__ == "__main__":
    for game_name in PARAS.keys():
        if game_name != "seesaw":
            continue
        print(game_name)
        vectors_file = f"./dataset1/game_seq_data/{game_name}/3/vectors.npy"  # Replace with your file path
        output_folder = "rendered_images1"  # Replace with your desired output folder
        vectors = np.load(vectors_file)
        print(vectors.shape)
        save_tensor_as_csv(torch.tensor(vectors),f"{game_name}_3.csv")
        vis_gif(vectors, output_folder,normalized=False,file_name=f"{game_name}_3.gif")