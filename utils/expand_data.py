import numpy as np

def expand_data(data, target_x, target_y, step_x=None, step_y=None):
    """
    Expand a 2D array to a specified size, filling the gaps with zeros and generating a mask.
    
    :param data: numpy 2D array, the original data
    :param target_x: int, the number of rows in the target array
    :param target_y: int, the number of columns in the target array
    :return: tuple (expanded_data, mask), where:
             - expanded_data: numpy 2D array, the expanded array
             - mask: numpy 2D array, a mask with 1s at original data positions and 0s elsewhere
    """
    original_x, original_y = data.shape
    if target_x < original_x or target_y < original_y:
        raise ValueError("Target size must be greater than or equal to the original size")

    # Create target-sized arrays for data and mask
    expanded_data = np.zeros((target_x, target_y), dtype=data.dtype)
    mask = np.zeros((target_x, target_y), dtype=np.uint8)

    # Calculate step intervals for mapping original data
    if step_x is None:
        step_x = (target_x - 1) // (original_x - 1) if original_x > 1 else 1
    if step_y is None:
        step_y = (target_y - 1) // (original_y - 1) if original_y > 1 else 1

    # Use NumPy slicing to map original data and create the mask
    expanded_data[::step_x if step_x > 0 else 1, ::step_y if step_y > 0 else 1] = data
    mask[::step_x if step_x > 0 else 1, ::step_y if step_y > 0 else 1] = 1

    return expanded_data, mask