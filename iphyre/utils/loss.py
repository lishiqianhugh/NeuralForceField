import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, gamma=1e4, alpha=1e2, beta=0):
        super(WeightedMSELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
    def forward(self, input, target):
        input = input.clone()
        target = target.clone()

        mse = (input - target) ** 2

        res_input = (input[:, 1:, :] - input[:, :-1, :]) / 0.1
        res_target = (target[:, 1:, :] - target[:, :-1, :]) / 0.1
        res_loss = (res_input - res_target) ** 2

        res_res_input = (res_input[:, 1:, :] - res_input[:, :-1, :]) / 0.1
        res_res_target = (res_target[:, 1:, :] - res_target[:, :-1, :]) / 0.1
        res_res_loss = (res_res_input - res_res_target) ** 2

        mse *= self.gamma
        res_loss *= self.alpha
        res_res_loss *= self.beta
        loss = mse.mean() + res_loss.mean() + res_res_loss.mean()

        return loss, mse.mean(), res_loss.mean(), res_res_loss.mean()