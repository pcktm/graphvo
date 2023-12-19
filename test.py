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
            output = output[-2]
            gt = gt[-2]
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
        ResetToFirstNode(),
        T.RemoveDuplicatedEdges(),
        T.ToUndirected(),
        T.VirtualNode(),
    ]
)

GRAPH_LENGTH = 16
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
state = torch.load("./models_euler/model_after_6.pth")
model.load_state_dict(state)
model.eval()

USE_FILE = True

if not USE_FILE or not os.path.exists("data/predicted.npy"):
    predicted, ground_truth = test(model, eval_dataloader, device, -2)
    np.save("data/predicted.npy", np.array(predicted))
    np.save("data/ground_truth.npy", np.array(ground_truth))
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
else:
    predicted = np.load("data/predicted.npy")
    ground_truth = np.load("data/ground_truth.npy")

# now, it is an array of shape (num_samples, 6)
# i want to plot just the ground truth to validate the graphs\
# importantly its in x, y, z, a, b, c format where y is the height so I don't care for it in 2d odometry plots
# a, b, c are euler angles


# plt.plot(ground_truth[:, 0], ground_truth[:, 2], label="ground truth")
# now, predicted has to be odometried, by summing up the relative rotations and positions
def integrate_movement(node_positions):
    positions = node_positions[:, :3]
    rotations = node_positions[:, 3:]
    integrated_positions = np.zeros_like(positions)
    integrated_rotations = np.zeros_like(rotations)

    for i in range(len(positions)):
        if i == 0:
            integrated_positions[i] = positions[i]
            integrated_rotations[i] = rotations[i]
        else:
            prev_rot = integrated_rotations[i - 1]
            rotation_matrix = R.from_euler("xyz", prev_rot).as_matrix()

            # Apply the relative rotation to the current position
            relative_rot_position = rotation_matrix.dot(positions[i])

            # Add the odometry adjusted position to previous position
            integrated_positions[i] = (
                integrated_positions[i - 1] + relative_rot_position
            )

            # Add the relative rotation to the previous rotation
            integrated_rotations[i] = integrated_rotations[i - 1] + rotations[i]

    return np.concatenate((integrated_positions, integrated_rotations), axis=1)

integrated_gt = integrate_movement(ground_truth)
integrated_predicted = integrate_movement(predicted)

# plot just the positions of ground truth and predicted
plt.figure()
plt.plot(integrated_gt[:, 0], integrated_gt[:, 2], label="ground truth")
plt.plot(integrated_predicted[:, 0], integrated_predicted[:, 2], label="predicted")
plt.legend()
plt.show()