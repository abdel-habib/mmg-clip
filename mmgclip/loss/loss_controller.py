from .losses import CLIPLoss, AveragedMedicalCLIPLoss, MMGCLIPLoss

def create_loss(loss_name):
    """
    Returns the specified loss controller class based on the provided loss_name.

    Current available loss classes:
    1. 'CLIPLoss': Implementation of CLIP loss.
    2. 'AveragedMedicalCLIPLoss': Modified implementation of CLIP loss for medical usecase.

    Parameters:
    - loss_name (str): The name of the loss class. This class must be imported inside this file.

    Returns:
    - class: The corresponding loss class.

    Raises:
    - ValueError: If the specified network_name does not correspond to a valid class.
    """
    network_class = globals().get(loss_name, None)
    if network_class is None:
        raise ValueError(f"Invalid network_name: {loss_name}")
    return network_class