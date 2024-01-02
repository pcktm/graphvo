import os
from torch_geometric.loader import DataLoader, CachedLoader
from scipy.spatial.transform import Rotation as R
import torch_geometric.transforms as T
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import (
    KittiSequenceDataset,
    SequenceGraphDataset,
)
from loss import AllNodesLoss, JustLastNodeLoss, LastNodeShiftLoss
from model import GraphVO

from utils import NormalizeKITTIPose, RelativeShift, ResetToFirstNode

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


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
        ResetToFirstNode(),
        # RelativeShift(),
        T.ToUndirected(),
        T.AddRemainingSelfLoops(),
        T.RemoveDuplicatedEdges(),
        # T.GDC(),
        # T.VirtualNode(),
    ]
)

GRAPH_LENGTH = 8
BATCH_SIZE = 128

# dataset = 2

# seq = SequenceGraphDataset(
#     train_kitti_datasets[dataset],
#     stride=15,
#     transform=transform,
#     graph_length=8,
# )

# seq_unnormalized = SequenceGraphDataset(
#     train_kitti_datasets[dataset],
#     stride=15,
#     transform=None,
#     graph_length=8,
# )

# # get five random graphs
# idx = np.random.randint(0, len(seq) - 128, 5)
# graphs = [seq[i] for i in idx]
# graphs_unnormalized = [seq_unnormalized[i] for i in idx]
# plt.figure(figsize=(10, 5))
# for graph, graph_unnormalized, index in zip(graphs, graphs_unnormalized, idx):
#     # set aspect ratio to 1
#     plt.plot(
#         graph_unnormalized.y[:, 0], graph_unnormalized.y[:, 2], label=f"Ścieżka {index}"
#     )
#     plt.plot(graph.y[:, 0], graph.y[:, 2], label=f"Ścieżka {index} (po transformacji)")
#     # scatter beginnings of the graphs
#     plt.scatter(graph.y[0, 0], graph.y[0, 2], c="black")
#     plt.scatter(graph_unnormalized.y[0, 0], graph_unnormalized.y[0, 2], c="black")
#     # show grid
# plt.gca().set_aspect("equal")
# plt.grid()
# plt.tight_layout()
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("z")
# # plt.title("Przekształcenie reset_to_first_node")
# # plt.show()
# # save to file
# plt.savefig("reset_to_first_node.png")


graph_datasets = [
    SequenceGraphDataset(
        train_kitti_dataset,
        stride=stride,
        transform=transform,
        graph_length=GRAPH_LENGTH,
    )
    for train_kitti_dataset in train_kitti_datasets
    for stride in [15]
]

train_dataloader = DataLoader(
    torch.utils.data.ConcatDataset(graph_datasets),
    batch_size=BATCH_SIZE,
    num_workers=0 if __debug__ else 14,
    shuffle=True,
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

model = GraphVO().to(device)

# criterion = JustLastNodeLoss(
#     alpha=20, batch_size=BATCH_SIZE, graph_length=GRAPH_LENGTH
# ).to(device)

criterion = AllNodesLoss(alpha=0.8).to(device)
# criterion = LastNodeShiftLoss(alpha=20, batch_size=BATCH_SIZE, graph_length=GRAPH_LENGTH).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

EPOCHS = 100
SAVE_MODEL = True
SAVE_INTERVAL = 100
SAVE_DIR = "models_img_stride15"
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
