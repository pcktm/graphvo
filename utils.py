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


def integrate_motion(data):
    # Assume data shape is [num_samples, 6], where [:, :3] are positions and [:, 3:] are Euler angles
    positions = data[:, :3]
    rotations = data[:, 3:]

    # Prepare containers for new positions and rotations, copy first samples
    new_positions = np.empty_like(positions)
    new_rotations = np.empty_like(rotations)
    new_positions[0] = positions[0]
    new_rotations[0] = rotations[0]

    # Convert rotations to matrices for convenient computations
    rotation_matrices = np.array(
        [R.from_euler("xyz", euler_angles).as_matrix() for euler_angles in rotations]
    )

    # Apply accumulated transformations for each subsequent sample
    for i in range(1, data.shape[0]):
        # Calculate new position by rotating and translating the old one
        new_positions[i] = (
            new_positions[i - 1] + rotation_matrices[i - 1] @ positions[i]
        )

        # Calculate new rotation by multiplying old rotation with the new one
        new_rotations[i] = R.from_matrix(
            rotation_matrices[i - 1] @ rotation_matrices[i]
        ).as_euler("xyz")

    # Concatenate and return new positions and rotations
    return np.hstack((new_positions, new_rotations))


def create_global_path(outputs, node_idx=9):
    # Extract specified node from across all graphs
    nodes = outputs[:, node_idx, :]

    num_nodes = len(nodes)

    # Initialize the first pose
    global_node_poses = np.zeros((num_nodes, 6))
    global_node_poses[0] = nodes[0]

    for i in range(1, num_nodes):
        if i == 1:
            prev_global_pose = global_node_poses[i - 1]

            # Compute previous global rotation matrix
            prev_global_rotation_matrix = R.from_euler(
                "xyz", prev_global_pose[3:]
            ).as_matrix()

            # update the global position
            global_node_poses[i, :3] = prev_global_pose[:3] + (
                prev_global_rotation_matrix @ nodes[i, :3]
            )

            # update the global rotation
            local_rotation_matrix = R.from_euler("xyz", nodes[i, 3:]).as_matrix()
            global_rotation_matrix = prev_global_rotation_matrix @ local_rotation_matrix

        else:
            direction_vector = (
                global_node_poses[i - 1, :3] - global_node_poses[i - 2, :3]
            )
            direction_angle = np.arctan2(direction_vector[2], direction_vector[0])

            # Apply rotation along y-axis to align direction vector with x-axis
            rot_mat = R.from_euler("y", -direction_angle).as_matrix()

            # update the global position to move along the previous direction vector
            global_node_poses[i, :3] = (
                global_node_poses[i - 1, :3] + rot_mat @ nodes[i, :3]
            )

            # update the global rotation
            local_rotation_matrix = R.from_euler("xyz", nodes[i, 3:]).as_matrix()
            global_rotation_matrix = (
                rot_mat @ prev_global_rotation_matrix @ local_rotation_matrix
            )

        global_node_poses[i, 3:] = R.from_matrix(global_rotation_matrix).as_euler("xyz")

    return global_node_poses
