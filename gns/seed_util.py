# seed_util.py
import os
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_seed(seed=0):
    # Set the environment variable for the seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['GLOBAL_SEED'] = str(seed)
    
    # Set the seed
    set_seed(seed)

def apply_seed():
    # Get the seed from the environment variable
    seed = int(os.environ.get('GLOBAL_SEED', 0))
    
    # Set the seed
    set_seed(seed)
