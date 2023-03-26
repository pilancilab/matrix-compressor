import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import default_conv_kwargs, hyp

#############################################
#            Network Components             #
#############################################


# We might be able to fuse this weight and save some memory/runtime/etc, since the fast version of the network might be able to do without somehow....
class BatchNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-12,
        momentum=hyp["net"]["batch_norm_momentum"],
        weight=False,
        bias=True,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


# Allows us to set default arguments for the whole convolution itself.
# Having an outer class like this does add space and complexity but offers us
# a ton of freedom when it comes to hacking in unique functionality for each layer type
class Conv(nn.Conv2d):
    def __init__(self, *args, norm=False, **kwargs):
        kwargs = {**default_conv_kwargs, **kwargs}
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.norm = norm

    def forward(self, x):
        if self.training and self.norm:
            # TODO: Do/should we always normalize along dimension 1 of the weight vector(s), or the height x width dims too?
            with torch.no_grad():
                F.normalize(self.weight.data, p=self.norm)
        return super().forward(x)


class Linear(nn.Linear):
    def __init__(self, *args, norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.norm = norm

    def forward(self, x):
        if self.training and self.norm:
            # TODO: Normalize on dim 1 or dim 0 for this guy?
            with torch.no_grad():
                F.normalize(self.weight.data, p=self.norm)
        return super().forward(x)


# can hack any changes to each residual group that you want directly in here
class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = Conv(channels_in, channels_out)
        self.conv2 = Conv(channels_out, channels_out)

        self.norm1 = BatchNorm(channels_out)
        self.norm2 = BatchNorm(channels_out)

        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        residual = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = x + residual  # haiku
        return x


class TemperatureScaler(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.scaler = torch.tensor(init_val)

    def forward(self, x):
        return x.mul(self.scaler)


class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2, 3))  # Global maximum pooling


class SpeedyResNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict  # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.net_dict["initial_block"]["whiten"](x)
        x = self.net_dict["initial_block"]["project"](x)
        x = self.net_dict["initial_block"]["activation"](x)
        x = self.net_dict["residual1"](x)
        x = self.net_dict["residual2"](x)
        x = self.net_dict["residual3"](x)
        x = self.net_dict["pooling"](x)
        x = self.net_dict["linear"](x)
        x = self.net_dict["temperature"](x)
        if not self.training:
            # Average the predictions from the lr-flipped inputs during eval
            orig, flipped = x.split(x.shape[0] // 2, dim=0)
            x = 0.5 * orig + 0.5 * flipped
        return x
