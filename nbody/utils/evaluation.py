import torch
import torch.nn as nn

class MeanAbsoluteError(nn.Module):
    def __init__(self):
        super(MeanAbsoluteError, self).__init__()

    def forward(self, input, target, dynamic_mask):

        abs_diff = torch.abs(input - target)
        dynamic_mask = dynamic_mask.expand_as(input)
        masked_abs_diff = abs_diff * dynamic_mask
        
        sum_abs_diff = masked_abs_diff.reshape(masked_abs_diff.size(0), -1).sum(dim=1)  # [batch_size]

        count = dynamic_mask.reshape(dynamic_mask.size(0), -1).sum(dim=1)  # [batch_size]

        mean_abs_diff = sum_abs_diff / (count)  # [batch_size]

        return mean_abs_diff

class MeanSquaredError(nn.Module):
    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def forward(self, input, target, dynamic_mask):
        mse = (input - target) ** 2
        dynamic_mask = dynamic_mask.expand_as(mse)
        masked_mse = mse * dynamic_mask
        sum_mse = masked_mse.reshape(masked_mse.size(0), -1).sum(dim=1)  # [batch_size]
        count = dynamic_mask.reshape(dynamic_mask.size(0), -1).sum(dim=1)  # [batch_size]
        mean_mse = sum_mse / count  # [batch_size]
        return mean_mse

class RootMeanSquaredError(nn.Module):
    def __init__(self):
        super(RootMeanSquaredError, self).__init__()

    def forward(self, input, target, dynamic_mask):
        mse = (input - target) ** 2
        dynamic_mask = dynamic_mask.expand_as(mse)
        masked_mse = mse * dynamic_mask
        sum_mse = masked_mse.reshape(masked_mse.size(0), -1).sum(dim=1)  # [batch_size]
        count = dynamic_mask.reshape(dynamic_mask.size(0), -1).sum(dim=1)  # [batch_size]
        mean_mse = sum_mse / count  # [batch_size]
        rmse = torch.sqrt(mean_mse)
        return rmse

class PearsonCorrelationCoefficient(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PearsonCorrelationCoefficient, self).__init__()
        self.epsilon = epsilon  # To prevent division by zero

    def forward(self, input, target, dynamic_mask):
        
        mask = dynamic_mask.expand_as(input)
        
        masked_input = input * mask  # [batch_size, steps, num_obj, 3]
        masked_target = target * mask  # [batch_size, steps, num_obj, 3]
        
        # Flatten the tensors per batch
        batch_size = input.size(0)
        masked_input = masked_input.reshape(batch_size, -1)  # [batch_size, steps*num_obj*3]
        masked_target = masked_target.reshape(batch_size, -1)  # [batch_size, steps*num_obj*3]
        mask_flat = mask.reshape(batch_size, -1)  # [batch_size, steps*num_obj*3]
        
        valid_counts = mask_flat.sum(dim=1)  # [batch_size]
        
        mean_input = masked_input.sum(dim=1) / valid_counts  # [batch_size]
        mean_target = masked_target.sum(dim=1) / valid_counts  # [batch_size]
        
        input_dev = masked_input - mean_input.unsqueeze(1)  # [batch_size, N]
        target_dev = masked_target - mean_target.unsqueeze(1)  # [batch_size, N]
        
        covariance = (input_dev * target_dev).sum(dim=1) / valid_counts  # [batch_size]
        variance_input = (input_dev ** 2).sum(dim=1) / valid_counts  # [batch_size]
        variance_target = (target_dev ** 2).sum(dim=1) / valid_counts  # [batch_size]
        
        pcc = covariance / (torch.sqrt(variance_input) * torch.sqrt(variance_target) + self.epsilon)  # [batch_size]
        
        pcc = torch.clamp(pcc, -1.0, 1.0)
        
        return pcc


class FinalPositionError(nn.Module):
    def __init__(self):
        super(FinalPositionError, self).__init__()

    def forward(self, input, target, dynamic_mask):
        # Get the last step
        mask_sum = dynamic_mask.sum(dim=[2, 3])  # [batch_size, steps]
        last_nonzero_step = (mask_sum != 0).sum(dim=1) - 1 # [batch_size]
        input_final = input[:, last_nonzero_step, :, :]  # [batch_size, num_obj, 3]
        target_final = target[:, last_nonzero_step, :, :]  # [batch_size, num_obj, 3]
        mask_final = dynamic_mask[:, last_nonzero_step, :, :]  # [batch_size, num_obj, 1]

        abs_diff = torch.abs(input_final - target_final)  # [batch_size, num_obj, 3]
        mask_final = mask_final.expand_as(abs_diff)
        masked_abs_diff = abs_diff * mask_final  # [batch_size, num_obj, 3]
        
        sum_abs_diff = masked_abs_diff.reshape(masked_abs_diff.size(0), -1).sum(dim=1)  # [batch_size]
        count = mask_final.reshape(mask_final.size(0), -1).sum(dim=1)  # [batch_size]
        
        mean_abs_diff = sum_abs_diff / count  # [batch_size]
        return mean_abs_diff

class PositionChangeError(nn.Module):
    def __init__(self):
        super(PositionChangeError, self).__init__()

    def forward(self, input, target, dynamic_mask):
        # Compute position change
        input_change = input[:, 1:, :, :] - input[:, :-1, :, :]  # [batch_size, steps-1, num_obj, 3]
        target_change = target[:, 1:, :, :] - target[:, :-1, :, :]  # [batch_size, steps-1, num_obj, 3]
        mask_change = dynamic_mask[:, 1:, :, :]  # [batch_size, steps-1, num_obj, 1]
        
        abs_diff = torch.abs(input_change - target_change)  # [batch_size, steps-1, num_obj, 3]
        mask_change = mask_change.expand_as(abs_diff)
        masked_abs_diff = abs_diff * mask_change  # [batch_size, steps-1, num_obj, 3]
        
        sum_abs_diff = masked_abs_diff.reshape(masked_abs_diff.size(0), -1).sum(dim=1)  # [batch_size]
        count = mask_change.reshape(mask_change.size(0), -1).sum(dim=1)  # [batch_size]
        
        mean_abs_diff = sum_abs_diff / count  # [batch_size]
        return mean_abs_diff