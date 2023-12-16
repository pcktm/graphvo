import torch
import torch.utils.data.dataset as dataset
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Data
import pykitti
import numpy as np
import os
from typing import Union, List
from torchvision.transforms import v2 as tv2


class KittiSequenceDataset(dataset.Dataset):
    def __init__(
        self,
        basedir,
        sequence_name,
        transform_image=None,
        transform_sample=None,
        load_images=True,
        return_rich_sample=False,
    ) -> None:
        super().__init__()
        self.basedir = basedir
        self.sequence_name = sequence_name
        self.transform_image = transform_image
        self.transform_sample = transform_sample
        self.load_images = load_images
        self.return_rich_sample = return_rich_sample
        self.sequence = pykitti.odometry(basedir, sequence_name)
        self.timestamps = self.sequence.timestamps
        self.num_samples = len(self.timestamps)
        self.default_transform = tv2.Compose(
            [
                tv2.PILToTensor(),
                tv2.Resize(size=(64, 64), antialias=True),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int | torch.Tensor | slice):
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, list):
            return [self[i] for i in index]

        image = self.sequence.get_cam2(index) if self.load_images else None
        image = self.default_transform(image) if image is not None else None

        try:
            pose = self.sequence.poses[index]
        except IndexError as e:
            print(f"Index {index} out of range for sequence {self.sequence_name}")
            raise e

        if self.transform_image is not None and image is not None:
            image = self.transform_image(image)

        if self.transform_sample is not None:
            pose = self.transform_sample(pose)

        if self.return_rich_sample:
            return image, pose, self.timestamps[index]

        return image, torch.tensor(pose, dtype=torch.float32)


class SequenceGraphDataset(dataset.Dataset):
    def __init__(
        self,
        base_dataset: KittiSequenceDataset,
        graph_length=5,
        transform=None,
    ) -> None:
        super().__init__()
        self.dataset = base_dataset
        self.graph_length = graph_length
        self.transform = transform

    def __getitem__(self, index):
        """
        Returns a graph of length self.graph_length, constructed by taking the frame at index as the last node
        and the previous self.graph_length frames as leading nodes.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        nodes = []
        y = []

        for i in range(self.graph_length - 1):
            node, label = self.dataset[index + i]
            nodes.append(node)
            y.append(label)

        # add the last node
        node, label = self.dataset[index + self.graph_length]
        nodes.append(node)
        y.append(label)

        # add edges, unidirected, all nodes are connected to each other
        edge_index = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        nodes = torch.stack(nodes)
        y = torch.stack(y)

        if self.transform:
            nodes, edge_index, y = self.transform(nodes, edge_index, y)

        return Data(x=nodes, edge_index=edge_index, y=y)

    def __len__(self):
        return self.dataset.__len__() - self.graph_length


class MultipleSequenceGraphDataset(dataset.Dataset):
    def __init__(
        self,
        sequences: List[KittiSequenceDataset],
        transform=None,
        graph_length=5,
    ) -> None:
        super().__init__()
        self.graph_length = graph_length
        self.datasets = [
            SequenceGraphDataset(
                sequence, graph_length=graph_length, transform=transform
            )
            for sequence in sequences
        ]

    def __getitem__(self, index):
        # find the dataset that contains the index and remember that index in that dataset should be local
        dataset_index = 0
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index][index]

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
