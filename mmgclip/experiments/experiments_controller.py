from .ClassifierExperiment import ClassifierExperiment as classification

def create_experiment(experiment_name):
    """
    Returns the specified experiment controller class based on the provided experiment_name.

    Current available experiments name:
    1. 'classification': For classification task using `ClassifierExperiment` class.

    Parameters:
    - experiment_class_name (str): The name of the neural network component/model. 
                                   This class must be imported inside this file.

    Returns:
    - class: The corresponding neural network class.

    Raises:
    - ValueError: If the specified network_name does not correspond to a valid class.
    """
    network_class = globals().get(experiment_name, None)
    if network_class is None:
        raise ValueError(f"Invalid network_name: {experiment_name}")
    return network_class