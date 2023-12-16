from scipy.spatial.transform import Rotation as R
import torch

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
