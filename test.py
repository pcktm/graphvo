import os
from torch_geometric.loader import DataLoader
import torch_geometric
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

from utils import NormalizeKITTIPose, RelativeShift, ResetToFirstNode

import matplotlib.pyplot as plt


def test(
    model: torch.nn.Module,
    test_loader,
    device,
    node_selector=None,
):
    model.eval()
    # every_16th_graph = test_loader.dataset[::16]
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

transform = T.Compose(
    [
        NormalizeKITTIPose(),
        RelativeShift(),
        T.ToUndirected(),
        T.AddRemainingSelfLoops(),
        T.RemoveDuplicatedEdges(),
        T.GDC(),
        T.VirtualNode(),
    ]
)

GRAPH_LENGTH = 24
BATCH_SIZE = 1

eval_dataset = SequenceGraphDataset(
    base_dataset=KittiSequenceDataset(basedir_kitti, "05"),
    graph_length=GRAPH_LENGTH,
    stride=7,
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
state = torch.load("./models_bitm_features_lg_stride/model_after_85.pth")
model.load_state_dict(state)
model.eval()

USE_FILE = False

if not USE_FILE or not os.path.exists("data/predicted.npy"):
    predicted, ground_truth = test(model, eval_dataloader, device)
    np.save("data/predicted.npy", np.array(predicted))
    np.save("data/ground_truth.npy", np.array(ground_truth))
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
else:
    predicted = np.load("data/predicted.npy")
    ground_truth = np.load("data/ground_truth.npy")

# from each graph select second node
node_idx = 12
gt_2 = ground_truth[:, node_idx]
integrated_gt = np.zeros_like(gt_2)
predicted_2 = predicted[:, node_idx]
integrated_predicted = np.zeros_like(predicted_2)

ip_3 = np.zeros_like(predicted_2)

for i in range(1, gt_2.shape[0]):
    integrated_gt[i] = integrated_gt[i - 1] + gt_2[i]
    integrated_predicted[i] = integrated_predicted[i - 1] + predicted_2[i]
    ip_3[i] = ip_3[i - 1] + predicted[:, 3][i]

plt.figure()
# plt.axes().set_aspect('equal', 'datalim')
plt.plot(integrated_gt[:, 0], integrated_gt[:, 2], label="ground truth")
plt.plot(integrated_predicted[:, 0], integrated_predicted[:, 2], label="predicted")
plt.plot(ip_3[:, 0], ip_3[:, 2], label="predicted 3")

# plot lines between points in gt and predicted
for i in range(0, gt_2.shape[0], 10):
    plt.plot(
        [integrated_gt[i, 0], integrated_predicted[i, 0]],
        [integrated_gt[i, 2], integrated_predicted[i, 2]],
        "g",
        lw=0.1,
    )

plt.legend()
plt.show()
