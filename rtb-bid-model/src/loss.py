import torch
import torch.nn as nn


class NLLLoss(nn.Module):
    # negative log-likelihood loss for a log-normal distribution
    # formula: L = log(sigma) + (log(p) - mu)^2 / (2 * sigma^2)

    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, log_target):
        # clamp sigma on both ends -- too small causes division to blow up,
        # too large means the model is just saying "I have no idea" and not learning anything
        sigma = sigma.clamp(min=1e-3, max=10.0)
        loss = torch.log(sigma) + (log_target - mu) ** 2 / (2 * sigma ** 2)
        return loss.mean()


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, log_target):
        return self.mse(pred, log_target)
