import numpy as np
import torch
from torch import nn

def atanh(x, eps=1e-6):
    # The inverse hyperbolic tangent function, missing in pytorch.
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))

def damping(msg_prev, msg_new, gamma):
    # Convex combination of new value and value from last iteration
    return (1 - gamma) * msg_prev + gamma * msg_new

## test
## test
## test
# test