from typing import Union

import torch
from torch import Tensor
import numpy as np
from scipy.spatial.transform import Rotation as R

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform("reset_to_first_node")
class ResetToFirstNode(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.stores:
            if "y" in store:
                y = store.y.detach().numpy()  # convert torch tensor to numpy array
                positions = y[:, :3]
                rotations = y[:, 3:].reshape(-1, 3, 3)

                # get reference position and rotation
                first_position = positions[0]
                second_position = positions[1]
                direction_vector = second_position - first_position
                direction_vector /= np.linalg.norm(direction_vector)

                # calculate rotation matrix to align the first two points along the x-axis (East)
                target_vector = np.array([1, 0, 0])
                v = np.cross(direction_vector, target_vector)
                s = np.linalg.norm(v)
                c = np.dot(direction_vector, target_vector)

                Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + Vx + Vx @ Vx * ((1 - c) / (s**2))

                # subtract first position from all positions to move the first node to the origin
                positions -= first_position

                for i in range(positions.shape[0]):
                    # apply rotation to positions and rotations about the origin
                    positions[i] = rotation_matrix @ positions[i]
                    rotations[i] = rotation_matrix @ rotations[i]

                # convert numpy arrays back to torch tensors
                positions = torch.from_numpy(positions).float()
                rotations = torch.from_numpy(
                    np.stack([R.from_matrix(r).as_euler("xyz") for r in rotations])
                ).float()

                # over-write 'y' attribute in the store
                store.y = torch.cat((positions, rotations), dim=1)
        return data


@functional_transform("relative_shift")
class RelativeShift(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.stores:
            if "y" in store:
                y = store.y.detach().numpy()

                positions = y[:, :3]
                rotations = y[:, 3:].reshape(-1, 3, 3)

                rel_positions = np.zeros_like(positions)
                rel_rotations = np.zeros_like(rotations)

                for i in range(1, positions.shape[0]):
                    prev_position = positions[i - 1]
                    prev_rotation = rotations[i - 1]
                    curr_position = positions[i]
                    curr_rotation = rotations[i]

                    # calculate relative position and rotation
                    rel_position = curr_position - prev_position
                    rel_rotation = prev_rotation.T @ curr_rotation

                    rel_positions[i] = rel_position                    
                    rel_rotations[i] = rel_rotation


                rel_positions = torch.from_numpy(rel_positions).float()
                rel_rotations = torch.from_numpy(
                    np.stack([R.from_matrix(r).as_quat() for r in rel_rotations])
                ).float()

                store.y = torch.cat((rel_positions, rel_rotations), dim=1)
        return data


@functional_transform("normalize_kitti_pose")
class NormalizeKITTIPose(BaseTransform):
    def __init__(self) -> None:
        super().__init__()
        # Values taken from https://github.com/aofrancani/TSformer-VO/blob/main/datasets/kitti.py
        self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.stores:
            if "y" in store:
                y = store.y.detach().numpy()

                positions = y[:, :3]
                rotations = y[:, 3:].reshape(-1, 3, 3)
                angles = np.stack([R.from_matrix(r).as_euler("zxy") for r in rotations])

                angles = (angles - self.mean_angles) / self.std_angles
                positions = (positions - self.mean_t) / self.std_t

                angles = np.stack(
                    [R.from_euler("zxy", a).as_matrix().flatten() for a in angles]
                )

                positions = torch.from_numpy(positions).float()
                angles = torch.from_numpy(angles).float()

                store.y = torch.cat((positions, angles), dim=1)
        return data
