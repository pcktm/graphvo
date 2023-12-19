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
from loss import AllNodesLoss, JustLastNodeLoss
from model import GraphVO

from torch_geometric.utils import dropout_edge, dropout_node

from utils import ResetToFirstNode

import matplotlib.pyplot as plt


def test(
    model: torch.nn.Module,
    test_loader,
    device,
    node_selector=None,
):
    model.eval()
    progress_bar = tqdm(test_loader, desc=f"Test", total=len(test_loader))
    predicted = []
    ground_truth = []
    for batch_idx, data in enumerate(progress_bar):
        data = data.to(device)
        output = model(data.x, data.edge_index).detach().cpu().numpy()
        gt = data.y.detach().cpu().numpy()
        if node_selector is not None:
            output = output[-2] - output[-3]
            gt = gt[-2] - gt[-3]
        predicted.append(output)
        ground_truth.append(gt)
    return predicted, ground_truth


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
        T.RemoveDuplicatedEdges(),
        T.ToUndirected(),
        T.VirtualNode(),
    ]
)

GRAPH_LENGTH = 35
BATCH_SIZE = 1

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
state = torch.load("./models_reset_to_first_node/model_after_32.pth")
model.load_state_dict(state)
model.eval()

USE_FILE = True

if not USE_FILE or not os.path.exists("data/predicted.npy"):
    predicted, ground_truth = test(model, eval_dataloader, device, -2)
    np.save("data/predicted.npy", np.array(predicted))
    np.save("data/ground_truth.npy", np.array(ground_truth))
else:
    predicted = np.load("data/predicted.npy")
    ground_truth = np.load("data/ground_truth.npy")

# now, it is an array of shape (num_samples, 7)
# i want to plot just the ground truth to validate the graphs\
# importantly its in x, y, z, w, x, y, z format where y is the height so I don't care for it in 2d odometry plots

# plt.plot(ground_truth[:, 0], ground_truth[:, 2], label="ground truth")
# now, predicted has to be odometried, by summing up the relative rotations and positions
predicted_odometried = np.zeros_like(predicted)
predicted_odometried[0] = [0, 0, 0, 1, 0, 0, 0]
for i in range(1, len(predicted)):
    predicted_odometried[i, :3] = (
        predicted_odometried[i - 1, :3] + predicted[i, :3]
    )  # add positions
    predicted_odometried[i, 3:] = (
        R.from_quat(predicted_odometried[i - 1, 3:])
        * R.from_quat(predicted[i, 3:])
    ).as_quat()  # add rotations

gt_odometried = np.zeros_like(ground_truth)
gt_odometried[0] = [0, 0, 0, 1, 0, 0, 0]
for i in range(1, len(ground_truth)):
    gt_odometried[i, :3] = (
        gt_odometried[i - 1, :3] + ground_truth[i, :3]
    )  # add positions
    gt_odometried[i, 3:] = (
        R.from_quat(gt_odometried[i - 1, 3:]) * R.from_quat(ground_truth[i, 3:])
    ).as_quat()  # add rotations

plt.plot(predicted_odometried[:, 0], predicted_odometried[:, 2], label="predicted")
plt.plot(gt_odometried[:, 0], gt_odometried[:, 2], label="ground truth")
plt.legend()
plt.show()