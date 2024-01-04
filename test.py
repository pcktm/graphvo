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

import matplotlib


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


print(matplotlib.rcParams.keys())
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{polski}",
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[polish]{babel}",
            ]
        ),
    }
)
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
        #ResetToFirstNode(),
        RelativeShift(),
        T.ToUndirected(),
        T.AddRemainingSelfLoops(),
        T.RemoveDuplicatedEdges(),
        # T.GDC(),
        # T.VirtualNode(),
    ]
)

GRAPH_LENGTH = 8
BATCH_SIZE = 1

eval_dataset = SequenceGraphDataset(
    base_dataset=KittiSequenceDataset(basedir_kitti, "02"),
    graph_length=GRAPH_LENGTH,
    stride=15,
    transform=transform,
)

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0 if __debug__ else 14,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

model = GraphVO()
state = torch.load("./models_img_stride_nodeindex_round2/model_after_100.pth")
model.load_state_dict(state)
model.to(device)
model.eval()

USE_FILE = True

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
node_idx = 6
gt_2 = ground_truth[:, node_idx]
integrated_gt = np.zeros_like(gt_2)
predicted_2 = predicted[:, node_idx]
integrated_predicted = np.zeros_like(predicted_2)

ip_3 = np.zeros_like(predicted_2)

for i in range(1, gt_2.shape[0]):
    integrated_gt[i] = integrated_gt[i - 1] + gt_2[i]
    integrated_predicted[i] = integrated_predicted[i - 1] + predicted_2[i]
    ip_3[i] = ip_3[i - 1] + predicted[:, 1][i]

fig, ax = plt.subplots(1,1, figsize=set_size(455))
ax.plot(integrated_gt[:, 0], integrated_gt[:, 2], label=r"\textit{Ground truth}")
ax.plot(
    integrated_predicted[:, 0], integrated_predicted[:, 2], label="Przewidywana ścieżka"
)
ax.scatter(0, 0, label="Początek sekwencji", c="black")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()
fig.savefig("data/integrated-path.pgf")
fig.savefig("data/integrated-path.png")


# seq = SequenceGraphDataset(
#     KittiSequenceDataset(basedir_kitti, "02"),
#     stride=15,
#     transform=transform,
#     graph_length=25,
# )

# seq_unnormalized = SequenceGraphDataset(
#     KittiSequenceDataset(basedir_kitti, "02"),
#     stride=15,
#     transform=None,
#     graph_length=25,
# )

# # get five random graphs
# idx = np.random.randint(0, len(seq) - 128, 5)
# graphs = [seq[i] for i in idx]
# graphs_unnormalized = [seq_unnormalized[i] for i in idx]

# fig, ax = plt.subplots(1, 2, figsize=set_size(455))
# # more space between subplots
# fig.subplots_adjust(wspace=0.3)

# for graph, graph_unnormalized, (index, idx) in zip(
#     graphs, graphs_unnormalized, enumerate(idx)
# ):
#     index = index + 1
#     # set aspect ratio to 1
#     ax[0].plot(
#         graph_unnormalized.y[:, 0], graph_unnormalized.y[:, 2], label=f"Ścieżka {index}"
#     )
#     ax[1].plot(graph.y[:, 0], graph.y[:, 2], label=f"Ścieżka {index} (po transformacji)")
#     # scatter beginnings of the graphs
#     ax[1].scatter(graph.y[0, 0], graph.y[0, 2], c="black")
#     ax[0].scatter(graph_unnormalized.y[0, 0], graph_unnormalized.y[0, 2], c="black")
#     # show grid
# ax[0].grid()
# ax[1].grid()
# # set labels
# ax[0].set_xlabel("x")
# ax[0].set_ylabel("y")
# ax[1].set_xlabel("x")
# ax[1].set_ylabel("y")

# # set aspect ratio to 1
# ax[0].set_aspect("equal", "datalim")
# ax[1].set_aspect("equal", "datalim")
# # titles
# ax[0].set_title("Przed transformacją")
# ax[1].set_title("Po transformacji")
# # save as pgf
# plt.savefig("data/graph-normalization.pgf")
# plt.savefig("data/graph-normalization.png")
