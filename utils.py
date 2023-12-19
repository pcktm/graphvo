from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from typing import Union

import torch
from torch import Tensor
import numpy as np

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


def extract_position_rotation(transform):
    """
    Extract position and rotation from a 4x4 transformation matrix.

    Args:
        transform (numpy.ndarray): a 4x4 transformation matrix.

    Returns:
        dict: a dictionary with keys 'position' and 'rotation', where
              'position' is a 3D vector representing the position and
              'rotation' is a quaternion representing the rotation.
    """

    assert transform.shape == (4, 4), "Input should be a 4x4 matrix."

    # Position is the last column of the matrix
    position = transform[:3, 3]

    # Rotation matrix is the first 3 columns and rows
    rotation_matrix = transform[:3, :3]

    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(rotation_matrix)

    return {"position": position, "rotation": rotation}


@functional_transform("reset_to_first_node")
class ResetToFirstNode(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if "y" not in store:
                continue
            y = store.y.numpy()
            positions = y[:, :3]
            rotations = [Quaternion(x) for x in y[:, 3:]]

            first_position = positions[0]
            positions = positions - first_position

            first_rotation = rotations[0]
            inverse_first_rotation = first_rotation.inverse

            for i in range(1, len(positions)):
                # rotate positions with respect to first node's rotation
                positions[i] = inverse_first_rotation.rotate(positions[i])
                rotations[i] = (
                    inverse_first_rotation * rotations[i]
                )  # apply relative rotation
                
            positions = torch.tensor(positions, dtype=torch.float32)
            rotations = torch.tensor(
                np.array([x.elements for x in rotations]), dtype=torch.float32
            )

            store.y = torch.cat((positions, rotations), dim=1)

        return data
