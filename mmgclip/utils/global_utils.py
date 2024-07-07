import os
import random
import numpy as np
import torch
from .logger import logger

def seeding(seed):
    '''Sets the seed for reproducibility.

    Args:
        seed (int): The seed to use for reproducibility.

    Returns:
        None
    '''
    logger.info(f"Seed = {seed}.")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.Generator().manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

def create_directory_if_not_exists(path):
    '''Creates a directory or multiple nested directories if they don't exist.
    
    Args:
        path (str): The path to the directory to create.

    Returns:
        None    
    '''
    if path is None:
        raise ValueError('Invalid path passed.')
    
    if not os.path.exists(path):
        os.makedirs(path)

    return str(path)