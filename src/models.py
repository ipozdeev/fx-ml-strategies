import torch
from torch import nn
import torch.nn.functional as F


def normalized_softmax(_x, dim=-1, temp=1.5):
    _res = (
        F.softmax(
           (_x - torch.mean(_x, dim=dim, keepdim=True)) / \
           torch.std(_x, dim=dim, keepdim=True) * \
           temp,
           dim=dim
        ) - 1 / _x.shape[-1]
    ) * 2
    return _res


class ConvNet(nn.Module):
    """
    Parameters
    ----------
    lookback : int
    n_tenors : int
        second (column) dimension of one term structure history
    init_weights : str
        'normal' to init as a N(0, 0.01) rv, 'carry' to init as the carry filter
    """
    def __init__(self, lookback, n_tenors, init_weights="normal"):
        super().__init__()

        # layers
        self.conv_1 = nn.Conv2d(1, 1, (lookback, n_tenors), n_tenors)

        # init weights
        with torch.no_grad():
            if init_weights == "normal":
                nn.init.normal_(self.conv_1.weight, std=0.01)
                self.conv_1.weight[0, 0, -1, 0] = 0.0
            else:
                # init weights are the 1-m carry filter
                w_0_carry = torch.zeros((1, 1, lookback, n_tenors)).float()
                w_0_carry[:, :, -1, 1] = -1.0
                self.conv_1.weight = nn.Parameter(w_0_carry)

            nn.init.zeros_(self.conv_1.bias)

        self.act = normalized_softmax

    def forward(self, x):
        # forward pass through layers
        x = self.conv_1(x)[:, 0, :, :]

        # activation
        x = self.act(x)

        return x


class ConvNetMaxPool(nn.Module):
    """DO NOT USE."""
    def __init__(self, lookback, n_tenors):
        super().__init__()

        self.lookback = lookback
        self.n_tenors = n_tenors
        self.ks = 2

        # layers
        self.conv_1 = nn.Conv2d(1, 1,
                                (lookback // self.ks, n_tenors // self.ks),
                                n_tenors // self.ks)
        self.maxpool = nn.MaxPool3d(kernel_size=(self.ks, self.ks, 1))

        # init weights
        with torch.no_grad():
            nn.init.normal_(self.conv_1.weight, std=0.01)
            nn.init.zeros_(self.conv_1.bias)

        self.act = normalized_softmax

    def forward(self, x):
        # maxpool
        x = x.view(-1, self.lookback, self.n_tenors, x.shape[-1] // self.n_tenors)
        x = self.maxpool(x)
        x = x.view(-1, 1, self.lookback // self.ks, x.shape[-2] * x.shape[-1])

        # forward pass through layers of dim (batches, channels, lookback, assets*tenors)
        x = self.conv_1(x)[:, 0, :, :]

        # activation
        x = self.act(x)

        return x


class Conv3DNet(nn.Module):
    """
    Parameters
    ----------
    lookback : int
    n_tenors : int
        second (column) dimension of one term structure history
    """
    def __init__(self, lookback, n_assets, n_tenors):
        super().__init__()

        self.lookback = lookback
        self.n_assets = n_assets
        self.n_tenors = n_tenors
        self.ks = 3

        # layers
        self.conv_1 = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(self.ks, self.ks, 1),
            stride=1
        )
        self.conv_2 = nn.Conv2d(1, 1,
                                (lookback-self.ks+1, n_tenors-self.ks+1),
                                stride=n_tenors-self.ks+1)

        # init weights
        with torch.no_grad():
            nn.init.normal_(self.conv_1.weight, std=0.01)
            nn.init.normal_(self.conv_2.weight, std=0.01)
            self.conv_2.weight[0, 0, -1, 0] = 0.0
            nn.init.zeros_(self.conv_1.bias)
            nn.init.zeros_(self.conv_2.bias)

        self.act = normalized_softmax

    def forward(self, x):
        # forward pass through layers
        x = self.conv_1(x)[:, 0, :, :]

        x = x.view(*x.shape[:2], x.shape[2]*x.shape[3]).unsqueeze(1)

        x = self.conv_2(x)[:, 0, :, :]

        # activation
        x = self.act(x)

        return x
