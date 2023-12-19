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
                first_rotation = rotations[0]

                # subtract reference position from all positions
                positions -= first_position

                # calculate rotation matrix that will align the first rotation to identity
                # This rotation is the transposed of the first_rotation matrix, as rotation matrices are orthogonal.
                first_rotation_inverse = first_rotation.transpose()

                for i in range(positions.shape[0]):
                    # rotate positions with respect to the first node's rotation
                    positions[i] = first_rotation_inverse @ positions[i]

                    # rotate rotations w.r.t. first node's rotation
                    rotations[i] = first_rotation_inverse @ rotations[i]

                # convert numpy arrays back to torch tensors
                positions = torch.from_numpy(positions).float()
                rotations = torch.from_numpy(
                    np.stack([R.from_matrix(r).as_euler("xyz") for r in rotations])
                ).float()

                # over-write 'y' attribute in the store
                store.y = torch.cat((positions, rotations), dim=1)
        return data
