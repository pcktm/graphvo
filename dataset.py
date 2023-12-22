import torch
import torch.utils.data.dataset as dataset
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Data
import pykitti
import numpy as np
from typing import Union, List
from torchvision.transforms import v2 as tv2
import os


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
                tv2.Grayscale(num_output_channels=1),
                tv2.Resize(size=(64, 64), antialias=True),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.features = self.load_features()
        print(self.features.shape)
        self.loaded_items_cache = {}

    def __len__(self):
        return self.num_samples
    
    def load_features(self):
        features = []
        for i in range(self.num_samples):
            path = os.path.join("./data", "features_kitti_bitm", self.sequence_name, f"{i}.npy")
            if os.path.exists(path):
                features.append(np.load(path))
            else:
                features.append(None)
                print(f"File {path} does not exist")
        return np.array(features)


    def __getitem__(self, index: int | torch.Tensor | slice):
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, list):
            return [self[i] for i in index]

        if index in self.loaded_items_cache:
            return self.loaded_items_cache[index]

        # image = self.sequence.get_cam2(index) if self.load_images else None
        # image = self.default_transform(image) if image is not None else None

        features = self.features[index]

        try:
            pose = self.sequence.poses[index]
            pose = np.concatenate([pose[:3, 3], pose[:3, :3].ravel()])
            # rotate the entire sequence so that the first node is at the origin
            first = self.sequence.poses[0]
            first_position = first[:3, 3]
            first_rotation = first[:3, :3]
            position = pose[:3]
            rotation = pose[3:].reshape(3, 3)
            # rotation also has to be applied to the position
            position = np.dot(first_rotation.T, position - first_position)
            rotation = np.dot(first_rotation.T, rotation)
            pose = np.concatenate([position, rotation.ravel()])
        except IndexError as e:
            print(f"Index {index} out of range for sequence {self.sequence_name}")
            raise e

        if self.transform_image is not None and image is not None:
            image = self.transform_image(image)

        if self.transform_sample is not None:
            pose = self.transform_sample(pose)

        if self.return_rich_sample:
            return image, pose, self.timestamps[index]

        data = (torch.tensor(features, dtype=torch.float32), torch.tensor(pose, dtype=torch.float32))

        self.loaded_items_cache[index] = data

        return data


class SequenceGraphDataset(dataset.Dataset):
    def __init__(
        self,
        base_dataset: KittiSequenceDataset,
        graph_length=5,
        stride=1,
        transform=None,
    ) -> None:
        super().__init__()
        self.dataset = base_dataset
        self.graph_length = graph_length
        self.transform = transform
        self.stride = stride

    def __getitem__(self, index: int | torch.Tensor | slice):
        """
        Returns a graph of length self.graph_length, constructed by taking the frame at index as the last node
        and the previous self.graph_length frames as leading nodes.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        nodes = []
        y = []

        for i in range(self.graph_length - 1):
            node, label = self.dataset[index + i * self.stride]
            nodes.append(node)
            y.append(label)

        # add the last node
        node, label = self.dataset[index + (self.graph_length - 1) * self.stride]
        nodes.append(node)
        y.append(label)

        # add edges, each connected to the next two nodes
        edge_index = []
        for i in range(self.graph_length - 2):
            edge_index.append([i, i + 1])
            edge_index.append([i, i + 2])

        # add the last edge
        edge_index.append([self.graph_length - 2, self.graph_length - 1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        nodes = torch.stack(nodes)
        y = torch.stack(y)

        # to node data append the node index, with stride
        nodes = torch.cat(
            (
                nodes,
                torch.tensor(
                    [i * self.stride for i in range(self.graph_length)],
                    dtype=torch.float32,
                ).unsqueeze(1),
            ),
            dim=1,
        )

        data = Data(x=nodes, edge_index=edge_index, y=y)
        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return self.dataset.__len__() - (self.graph_length - 1) * self.stride


# TODO: use torch.utils.data.ConcatDataset
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

    def __getitem__(self, index: int | torch.Tensor | slice):
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        # find the dataset that contains the index and remember that index in that dataset should be local
        dataset_index = 0
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1

        return self.datasets[dataset_index][index]

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])


def WholeSequenceDataset(base_dataset: KittiSequenceDataset, transform=None):
    nodes = []
    y = []

    for i in range(len(base_dataset)):
        node, label = base_dataset[i]
        nodes.append(node)
        y.append(label)

    edge_index = []
    for i in range(len(base_dataset) - 1):
        edge_index.append([i, i + 1])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    nodes = torch.stack(nodes)
    y = torch.stack(y)

    data = Data(x=nodes, edge_index=edge_index, y=y)
    if transform is not None:
        data = transform(data)

    return data
