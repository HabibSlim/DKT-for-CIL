"""
Cosine linear module used in LUCIR.
"""
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module


class CosineLinear(Module):
    """
    This class implements the cosine normalizing linear layer module, following eq.4. of the LUCIR paper.
    """
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitializing network parameters (Xavier's initialization).
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, x):
        """
        Performing a forward pass.
        """
        out = F.linear(F.normalize(x, p=2, dim=1),
                       F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(Module):
    """
    Two FC layers, with outputs concatenated.
    """
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        """
        Performing a forward pass.
        """
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1)
        if self.sigma is not None:
            out = self.sigma * out
        return out
