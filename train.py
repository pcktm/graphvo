import os
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import KittiSequenceDataset, MultipleSequenceGraphDataset
from loss import JustLastNodeLoss
from model import GraphVO

from torch_geometric.utils import dropout_edge, dropout_node


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
    for batch_idx, data in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
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
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
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
        output = model(data)
        loss = criterion(output, data.y)
        loss_list.append(loss.item())
        progress_bar.set_postfix(loss=loss.item())
    return np.mean(loss_list)


os.makedirs("data", exist_ok=True)
basedir_kitti = "/home/pcktm/inzynierka/kitti/dataset"
train_sequences_kitti = ["00", "01", "02", "03", "04", "05", "06", "07", "08"]
train_kitti_datasets = [
    KittiSequenceDataset(
        basedir_kitti,
        sequence_name,
    )
    for sequence_name in train_sequences_kitti
]

GRAPH_LENGTH = 5
BATCH_SIZE = 64

train_dataset = MultipleSequenceGraphDataset(train_kitti_datasets, graph_length=5)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

model = GraphVO().to(device)
criterion = JustLastNodeLoss(
    batch_size=BATCH_SIZE, graph_length=GRAPH_LENGTH, alpha=20
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

EPOCHS = 100
SAVE_MODEL = True
SAVE_INTERVAL = 100
SAVE_DIR = "models"
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
