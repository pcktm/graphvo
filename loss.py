import torch
import torch.nn as nn
import torch.nn.functional as F


class JustLastNodeLoss(torch.nn.Module):
    def __init__(self, batch_size, graph_length, alpha = 20):
        super(JustLastNodeLoss, self).__init__()
        self.batch_size = batch_size
        self.graph_length = graph_length
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.view(-1, self.graph_length, 7)
        target = target.view(-1, self.graph_length, 7)

        # Get the last nodes for each element in the batch
        last_pred = pred[:, -1, :]
        last_target = target[:, -1, :]

        position_pred = last_pred[:, :3]
        position_target = last_target[:, :3]

        rotation_pred = last_pred[:, 3:]
        rotation_target = last_target[:, 3:]

        position_loss = F.mse_loss(position_pred, position_target)
        rotation_loss = F.mse_loss(rotation_pred, rotation_target)

        return position_loss + rotation_loss * self.alpha