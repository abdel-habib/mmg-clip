from .encoder import ResNet50Encoder, BertEncoder

def getNetworkClass(network_name):
    """
    Returns the specified neural network component/class based on the provided network_name.

    Parameters:
    - network_name (str): The name of the neural network component/model.

    Returns:
    - class: The corresponding neural network class.

    Raises:
    - ValueError: If the specified network_name does not correspond to a valid class.
    """
    network_class = globals().get(network_name, None)
    if network_class is None:
        raise ValueError(f"Invalid network_name: {network_name}")
    return network_class