import torch
import torch.nn as nn
import torch.nn.functional as F


class JustLastNodeLoss(torch.nn.Module):
    def __init__(self, batch_size, alpha=20, graph_length=5):
        super(JustLastNodeLoss, self).__init__()
        self.batch_size = batch_size
        self.graph_length = graph_length
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.shape == target.shape, "pred and target should have the same shape."
        assert pred.shape[1] == 7, "pred and target should have 7 columns."
        assert (
            self.batch_size * self.graph_length == pred.shape[0]
        ), "pred should have shape (batch_size * graph_length, 7)."

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


class LastNodeShiftLoss(torch.nn.Module):
    def __init__(self, batch_size, alpha=20, graph_length=5):
        super(LastNodeShiftLoss, self).__init__()
        self.batch_size = batch_size
        self.graph_length = graph_length
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.shape == target.shape, "pred and target should have the same shape."

        pred = pred.view(-1, self.graph_length, 6)
        target = target.view(-1, self.graph_length, 6)

        # Get the last nodes for each element in the batch
        last_pred = pred[:, -1, :]
        last_target = target[:, -1, :]

        second_last_pred = pred[:, -2, :]
        second_last_target = target[:, -2, :]

        position_pred = last_pred[:, :3]
        position_target = last_target[:, :3]

        rotation_pred = last_pred[:, 3:]
        rotation_target = last_target[:, 3:]

        second_last_position_pred = second_last_pred[:, :3]
        second_last_position_target = second_last_target[:, :3]

        second_last_rotation_pred = second_last_pred[:, 3:]
        second_last_rotation_target = second_last_target[:, 3:]

        position_loss = F.mse_loss(
            position_pred - second_last_position_pred,
            position_target - second_last_position_target,
        )
        rotation_loss = F.mse_loss(
            rotation_pred - second_last_rotation_pred,
            rotation_target - second_last_rotation_target,
        )

        return position_loss + rotation_loss * self.alpha


class AllNodesLoss(torch.nn.Module):
    def __init__(self, alpha=20):
        super(AllNodesLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        assert (
            pred.shape == target.shape
        ), f"pred and target should have the same shape, got {pred.shape} and {target.shape}."

        position_pred = pred[:, :3]
        position_target = target[:, :3]

        rotation_pred = pred[:, 3:]
        rotation_target = target[:, 3:]

        position_loss = F.mse_loss(position_pred, position_target, reduction="mean")
        rotation_loss = F.mse_loss(rotation_pred, rotation_target, reduction="mean")

        return position_loss + rotation_loss * self.alpha
