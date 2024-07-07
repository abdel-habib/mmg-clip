from .projection import LinearProjectionLayer, MLPProjectionHead, MultiLinearHead

def get_projection_head(projection_name):
    """
    Returns the specified projection controller class based on the provided projection_name.

    Current available loss classes:
    1. 'LinearProjectionLayer': A linear projection layer.
    2. 'MLPProjectionHead': Multi-layer Preceptron head.
    3. 'MultiLinearHead': Multiple linear projection head.

    Parameters:
    - projection_name (str): The name of the projection layer class. This class must be imported inside this file.

    Returns:
    - class: The corresponding projection class.

    Raises:
    - ValueError: If the specified network_name does not correspond to a valid class.
    """
    network_class = globals().get(projection_name, None)
    if network_class is None:
        raise ValueError(f"Invalid network_name: {projection_name}")
    return network_class