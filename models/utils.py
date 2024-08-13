import torch
from torch import nn
from torch.nn.utils import spectral_norm


def xavier_init_layer(
    in_size, out_size=None, spec_norm=True, layer_type=nn.Linear, **kwargs
):
    "Create a layer with spectral norm, xavier uniform init and zero bias"
    if out_size is None:
        out_size = in_size

    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)

    # Perform initialization
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)

    return layer

def get_model_param(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Learnable_sigmoid(nn.Module):
    """Implementation of a leanable sigmoid.

    Arguments
    ---------
    in_features : int
        Input dimensionality
    """

    def __init__(self, in_features=257):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True  # set requiresGrad to true!

        # self.scale = nn.Parameter(torch.ones(1))
        # self.scale.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        return 1.2 * torch.sigmoid(self.slope * x)

