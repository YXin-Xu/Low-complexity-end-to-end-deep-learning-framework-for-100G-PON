# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:29:46 2021

@author: second
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReLU2(nn.Module):
    def __init__(self):
        super(ReLU2, self).__init__()
    def forward(self, x):
        return F.relu(x) ** 2


class ReLU3(nn.Module):
    def __init__(self):
        super(ReLU3, self).__init__()
    def forward(self, x):
        return F.relu(x) ** 3
    

class sReLU(nn.Module):
    def __init__(self):
        super(sReLU, self).__init__()
    def forward(self, x):
        return F.relu(1-x) * F.relu(x)


class wave(nn.Module):
    def __init__(self):
        super(wave, self).__init__()
    def forward(self, x):
        return F.relu(x) - 2*F.relu(x-1/4) + 2*F.relu(x-3/4) -F.relu(x-1)
    
    
class phi2(nn.Module):
    def __init__(self):
        super(phi2, self).__init__()
    def forward(self, x):
        return F.relu(x) - 2*F.relu(x-1) + F.relu(x-2)
    

class s2ReLU(nn.Module):
    def __init__(self):
        super(s2ReLU, self).__init__()
    def forward(self, x):
        return torch.sin(2*np.pi*x) * sReLU(x)


    
