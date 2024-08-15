from speechbrain.utils.checkpoints import torch_recovery
import matplotlib
import matplotlib.pyplot as plt

from train import *
from kan_convs import KANConv2DLayer
from efficient_kan import KANLinear
from models.d4 import MetricDiscriminator

matplotlib.use('pgf')
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'


save_path = "../../thesis/figures/"
load_path = "../MetricGAN_KAN-d2/weights/"
samples = 400
figsize = (10, 5)

@torch.no_grad()
def bspline_shape(layer: KANLinear, x: torch.Tensor, scaled: bool = False):
    """
    Compute the bspline curve given layer and x.
    """
    assert x.size(-1) == layer.in_features
    x = x.reshape(-1, layer.in_features)

    spl_output = layer.b_splines(x)
    c = layer.spline_weight if not scaled else layer.spline_weight * layer.spline_scaler.unsqueeze(-1)

    spl_output = torch.einsum("abc,dbc->dab", c, spl_output)

    return spl_output

@torch.no_grad()
def base_activate_shape(layer: KANLinear, x: torch.Tensor, scaled: bool = True):
    assert x.size(-1) == layer.in_features
    return layer.base_activation(x) * layer.base_weight.unsqueeze(0) if scaled else layer.base_activation(x)

@torch.no_grad()
def kan_splines(layer: KANLinear, neurons=[0], range=[-1, 1], scaled: bool = False, samples=400):
    device = next(layer.parameters()).device
    x = torch.linspace(range[0], range[1], samples)
    x_in = x.unsqueeze(-1).unsqueeze(-1).expand(samples, layer.out_features, layer.in_features).to(device=device)
    y = bspline_shape(layer, x_in, scaled).detach().cpu()[:, neurons, :]
    g = layer.grid[neurons, layer.spline_order-1:-(layer.spline_order-1)].detach().cpu()
    c = layer.spline_weight[neurons, :, :].detach().cpu()

    return x, y, g, c

@torch.no_grad()
def kan_function_shape(layer: KANLinear, neurons=[0], range=[-4, 4], samples=400):
    device = next(layer.parameters()).device
    x = torch.linspace(range[0], range[1], samples)
    x_in = x.unsqueeze(-1).unsqueeze(-1).expand(samples, layer.out_features, layer.in_features).to(device=device)
    y = bspline_shape(layer, x_in, True) + base_activate_shape(layer, x_in)
    return x, y[:, neurons, :].detach().cpu()

@torch.no_grad()
def ckan_spline_shape(layer: KANConv2DLayer, x: torch.Tensor, idx=[0, 0]):
    in_idx = idx[0]
    out_idx = idx[1]
    in_group_size = layer.inputdim // layer.groups
    out_group_size = layer.outdim // layer.groups
    group_idx = out_idx // out_group_size
    assert group_idx == in_idx // in_group_size
    _in_idx = in_idx % in_group_size
    _out_idx = out_idx % out_group_size
    
    step = layer.grid_size + layer.spline_order
    left = _in_idx * step
    right = (_in_idx + 1) * step

    x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
    # Compute the basis for the spline using intervals and input values.
    target = x.shape[1:] + layer.grid.shape
    grid = layer.grid.view(*list([1 for _ in range(layer.ndim + 1)] + [-1, ])).expand(target).contiguous().to(
        x.device)

    bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)

    # Compute the spline basis over multiple orders.
    for k in range(1, layer.spline_order + 1):
        left_intervals = grid[..., :-(k + 1)]
        right_intervals = grid[..., k:-1]
        delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                            right_intervals - left_intervals)
        bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
    bases = bases.contiguous()[:, _in_idx, :, :] # .moveaxis(-1, 1) # B(x)
    weight = layer.spline_conv[group_idx].state_dict()['weight'][_out_idx, left : right, :, :].moveaxis(0, -1)

    spline_output = torch.einsum("abcd,bcd->abc", bases, weight).detach().cpu()
    weight = weight.detach().cpu()

    # print(group_idx, _out_idx, left, right)
    # print(bases.shape)
    # print(weight.shape)
    # print(grid.shape)
    return spline_output, grid[in_idx, :, :, layer.spline_order-1 : -(layer.spline_order-1)].detach().cpu(), weight


@torch.no_grad()
def ckan_splines(layer: KANConv2DLayer, idx=[0, 0], range=[-1, 1], samples=400):
    device = next(layer.parameters()).device
    x = torch.linspace(range[0], range[1], samples)
    x_in = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(samples, 2, *layer.kernel_size).to(device=device)
    return x, *ckan_spline_shape(layer, x_in, idx)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dis1 = MetricDiscriminator().to(device=device)
    dis2 = MetricDiscriminator().to(device=device)

    torch_recovery(dis1, load_path + "d-epoch350", True)
    torch_recovery(dis2, load_path + "d-epoch400", True)

    linear1 = dis1.Linear1
    linear2 = dis2.Linear1

    x, y, grid, c = kan_splines(linear1)
    _, y1, _, c1 = kan_splines(linear2)

    plt.figure(figsize=figsize)
    for i in range(1,16):
        # Plot curve
        ax = plt.subplot(5, 3, i)
        # Plot controlling points
        ax.scatter(grid, c[0, i-1, :], marker='x', linewidths=0.6)
        ax.scatter(grid, c1[0, i-1, :], marker='+', linewidths=0.6)
        ax.plot(x, y[:, 0, i-1], linewidth=0.6)
        ax.plot(x, y1[:, 0, i-1], linewidth=0.6)
        # plt.xlabel(f"({chr(ord('a')+i-1)})")
    
    plt.tight_layout(pad=0.2)
    plt.savefig(save_path + "spline_visualise.pdf", bbox_inches='tight')

    x, y = kan_function_shape(linear1)
    _, y1 = kan_function_shape(linear2)

    plt.figure(figsize=figsize)
    for i in range(1,16):
        # Plot curve
        plt.subplot(5, 3, i)
        # Plot controlling points
        plt.plot(x, y[:, 0, i-1], linewidth=0.6)
        plt.plot(x, y1[:, 0, i-1], linewidth=0.6)
        # plt.xlabel(f"({chr(ord('a')+i-1)}) $w_1={w1[i-1]:6.5f}$, $w_2={w2[i-1]:6.5f}$")
    
    plt.tight_layout(pad=0.2)
    plt.savefig(save_path + "phi_x.pdf", bbox_inches='tight')

    conv1 = dis1.conv3
    conv2 = dis2.conv3

    x, y, grid, w = ckan_splines(conv1, [1,2])
    _, y1, _, w1 = ckan_splines(conv2, [1,2])

    plt.figure(figsize=figsize)
    for i in range(5):
        for j in range(5):
            # Plot curve
            plt.subplot(5, 5, i*5+j+1)
            # Plot controlling points
            plt.plot(x, y[:, i, j], linewidth=0.6)
            plt.plot(x, y1[:, i, j], linewidth=0.6)
            plt.scatter(grid[i, j, :], w[i, j, :], marker='x', linewidths=0.6)
            plt.scatter(grid[i, j, :], w1[i, j, :], marker='+', linewidths=0.6)
            # plt.xlabel(f"({chr(ord('a')+i-1)}) $w_1={w1[i-1]:6.5f}$, $w_2={w2[i-1]:6.5f}$")

    plt.tight_layout(pad=0.2)
    plt.savefig(save_path + "conv_spline.pdf", bbox_inches='tight')
