"""
Defining the BiC and AdaptiveBiC linear layers.
"""
from enum import Enum
import torch


class CalibrationMethod(Enum):
    """
    Defining strategies for alpha/beta calibration.
    """
    FORWARD_LAST = 0,  # Training only last alpha/beta pair, applied only on last task classes
    FORWARD_ALL  = 1,  # Training all alpha/beta pairs, applied on all classes
    FORWARD_PAST = 2,  # Training last alpha/beta pair, applied on all past classes

    ADAPTIVE = 3,      # Training all alpha/beta pairs simultaneously, applied on all classes


class BiCLayer(torch.nn.Module):
    """
    Defining a BiC layer for a single task (2 parameters).
    """

    def __init__(self, device, init_zero=False):
        super(BiCLayer, self).__init__()

        if init_zero:
            self.alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device=device))
            self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device=device))
        else:
            self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False, device=device))
            self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device=device))

    def forward(self, x):
        """
        Overloading the forward pass.
        """
        return self.alpha * x + self.beta


class BiCNet(torch.nn.Module):
    """
    Extending the BiC layer for any number of tasks.
    """

    def __init__(self, device, method):
        super(BiCNet, self).__init__()

        self.device = device
        self.method = method
        self.t = 0

        # Initializing the local list of BiC "heads"
        self.bias_layers = torch.nn.ModuleList()

    def add_head(self):
        """
        Extending the layer for an additional task.
        """
        self.bias_layers += [BiCLayer(self.device).to(self.device)]
        self.t += 1

    def beta_l2(self, loss, lambd):
        """
        Adding an L2 loss over all trainable beta variables.
        """
        for layer in self.bias_layers:
            if layer.beta.requires_grad:
                loss += lambd * ((layer.beta[0] * layer.beta[0]) / 2.)
        return loss

    def forward(self, x):
        """
        Forwarding the BiC model using a specific method.
        """
        if self.method in [CalibrationMethod.FORWARD_ALL, CalibrationMethod.ADAPTIVE]:
            return self.forward_all(x)
        elif self.method == CalibrationMethod.FORWARD_LAST:
            return self.forward_last(x)
        elif self.method == CalibrationMethod.FORWARD_PAST:
            return self.forward_past(x)

    def forward_all(self, x):
        """
        Forwarding through every BiC layer.

        Args:
            x:      Logits extracted for all previous classes.
        """
        bic_outputs = []

        for i, x_ in enumerate(torch.chunk(x, self.t, dim=1)):
            bic_outputs += [self.bias_layers[i](x_)]

        return bic_outputs

    def forward_last(self, x):
        """
        Multiply all logits of the last task with the last alpha/beta coefficients learned.

        Args:
            x:     Logits extracted for all previous classes.
        """
        bic_outputs = []

        for i, x_ in enumerate(torch.chunk(x, self.t, dim=1)):
            if i == self.t - 1:
                bic_outputs += [self.bias_layers[i](x_)]
            else:
                bic_outputs += [x_]

        return bic_outputs

    def forward_past(self, x):
        """
        Multiply all logits of all past tasks with the last alpha/beta coefficients learned.

        Args:
            x:     Logits extracted for all previous classes.
        """
        bic_outputs = []

        for i, x_ in enumerate(torch.chunk(x, self.t, dim=1)):
            if i < self.t - 1:
                bic_outputs += [self.bias_layers[self.t - 1](x_)]
            else:
                bic_outputs += [x_]

        return bic_outputs

    def print_parameters(self, prec=5):
        """
        Printing model parameters.
        """
        max_chars = max(8, prec + 5)

        print("BiC params:")
        print("------------")
        for t, layer in enumerate(self.bias_layers):
            print("\tLayer %d" % t)
            print("\t\tα=", str(round(float(layer.alpha), prec)).ljust(max_chars), " | %s"
                  % ("active" if layer.alpha.requires_grad else "frozen"))
            print("\t\tβ=", str(round(float(layer.beta), prec)).ljust(max_chars), " | %s"
                  % ("active" if layer.alpha.requires_grad else "frozen"))
            print("\t------------")
        print("------------")

    def set_alpha(self, val):
        """
        Setting alpha coefficient value on the last layer.
        """
        self.bias_layers[-1].alpha = torch.nn.Parameter(torch.tensor(val, device=self.device, requires_grad=False))

    def set_beta(self, val):
        """
        Setting alpha coefficient value on the last layer.
        """
        self.bias_layers[-1].beta = torch.nn.Parameter(torch.tensor(val, device=self.device, requires_grad=False))

    def train_init(self):
        """
        Initializing training of the linear layer.
        """
        if self.method == CalibrationMethod.ADAPTIVE:
            self.train_all()
        else:
            self.train_last()

    def train_last(self):
        """
        Disabling the update of all 1..t-1 BiC heads (where t is the current head count, and the id of the last task).
        Enabling updates on the last layer.
        """
        for layer in self.bias_layers[:self.t - 1]:
            layer.alpha.requires_grad = False
            layer.beta.requires_grad = False

        self.bias_layers[self.t - 1].alpha.requires_grad = True
        self.bias_layers[self.t - 1].beta.requires_grad = True

    def train_all(self):
        """
        Enabling updates on the all BiC layers.
        """
        for layer in self.bias_layers:
            layer.alpha.requires_grad = True
            layer.beta.requires_grad = True

    def trainable_p(self):
        """
        Return the list of trainable parameters.
        """
        params = []
        for layer in self.bias_layers:
            if layer.alpha.requires_grad:
                params += [layer.alpha]
            if layer.beta.requires_grad:
                params += [layer.beta]
        return params
