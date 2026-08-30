import random
import numpy as np
import torch
import os

def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across standard library, numpy, and PyTorch.
    
    Note: Setting cudnn.deterministic = True may incur a slight performance cost 
    on the GPU, but it is necessary for exact reproducibility across runs.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
