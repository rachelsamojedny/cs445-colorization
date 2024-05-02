import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def loss_L2(ypred, ytrue):
    return np.linalg.norm(ypred, ytrue)

def loss(ypred, ytrue):
    ######
    return None