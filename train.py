import os
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation as R
import torch_geometric.transforms as T
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import (
    KittiSequenceDataset,
    MultipleSequenceGraphDataset,
    SequenceGraphDataset,
)
from loss import AllNodesLoss, JustLastNodeLoss, LastNodeShiftLoss
from model import GraphVO

from utils import NormalizeKITTIPose, RelativeShift, ResetToFirstNode

import matplotlib.pyplot as plt


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    save_model,
    save_interval,
    save_dir,
    scheduler=None,
):
    model.train()
    progress_bar = tqdm(
        train_loader, desc=f"Train Epoch {epoch}", total=len(train_loader)
    )
    loss_list = []
    for batch_idx, data in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        loss_list.append(loss.item())
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        progress_bar.set_postfix(loss=loss.item())
        if save_model and batch_idx % save_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"model_{epoch}_{batch_idx}.pth"),
            )
    if save_model:
        torch.save(
            model.state_dict(), os.path.join(save_dir, f"model_after_{epoch}.pth")
        )
    print(f"Train Epoch {epoch}: {np.mean(loss_list)}")
    return model


def evaluate(
    model,
    test_loader,
    criterion,
    device,
    epoch,
):
    model.eval()
    progress_bar = tqdm(test_loader, desc=f"Test Epoch {epoch}", total=len(test_loader))
    loss_list = []
    for batch_idx, data in enumerate(progress_bar):
        data = data.to(device)
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss_list.append(loss.item())
        progress_bar.set_postfix(loss=loss.item())
    return np.mean(loss_list)


os.makedirs("data", exist_ok=True)
basedir_kitti = "/home/pcktm/inzynierka/kitti/dataset"
train_sequences_kitti = ["00", "02", "08", "09"]
train_kitti_datasets = [
    KittiSequenceDataset(
        basedir_kitti,
        sequence_name,
    )
    for sequence_name in train_sequences_kitti
]

transform = T.Compose(
    [
        NormalizeKITTIPose(),
        RelativeShift(),
        T.RemoveDuplicatedEdges(),
        T.ToUndirected(),
        T.VirtualNode(),
    ]
)

GRAPH_LENGTH = 10
BATCH_SIZE = 64

train_dataset = MultipleSequenceGraphDataset(
    train_kitti_datasets, graph_length=GRAPH_LENGTH, transform=transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0 if __debug__ else 14,
)

eval_dataset = SequenceGraphDataset(
    base_dataset=KittiSequenceDataset(basedir_kitti, "05"),
    graph_length=GRAPH_LENGTH,
    transform=transform,
)

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0 if __debug__ else 14,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

model = GraphVO().to(device)

# criterion = JustLastNodeLoss(
#     alpha=20, batch_size=BATCH_SIZE, graph_length=GRAPH_LENGTH
# ).to(device)

criterion = AllNodesLoss(alpha=20).to(device)
# criterion = LastNodeShiftLoss(alpha=20, batch_size=BATCH_SIZE, graph_length=GRAPH_LENGTH).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

EPOCHS = 100
SAVE_MODEL = True
SAVE_INTERVAL = 100
SAVE_DIR = "models_relative_shift"
os.makedirs(SAVE_DIR, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model = train(
        model,
        train_dataloader,
        optimizer,
        criterion,
        device,
        epoch,
        SAVE_MODEL,
        SAVE_INTERVAL,
        SAVE_DIR,
    )
    eval_loss = evaluate(model, eval_dataloader, criterion, device, epoch)
    print(f"Eval loss: {eval_loss}")
