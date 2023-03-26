import torch
import torch.nn as nn
import copy

########################################
#          Training Helpers            #
########################################


class NetworkEMA(nn.Module):
    def __init__(self, net):
        super().__init__()  # init the parent module so this module is registered properly
        self.net_ema = copy.deepcopy(net).eval().requires_grad_(False)  # copy the model

    def update(self, current_net, decay):
        with torch.no_grad():
            for ema_net_parameter, (parameter_name, incoming_net_parameter) in zip(
                self.net_ema.state_dict().values(), current_net.state_dict().items()
            ):  # potential bug: assumes that the network architectures don't change during training (!!!!)
                if incoming_net_parameter.dtype in (torch.half, torch.float):
                    ema_net_parameter.mul_(decay).add_(
                        incoming_net_parameter.detach().mul(1.0 - decay)
                    )  # update the ema values in place, similar to how optimizer momentum is coded
                    # And then we also copy the parameters back to the network, similarly to the Lookahead optimizer (but with a much more aggressive-at-the-end schedule)
                    if (
                        not ("norm" in parameter_name and "weight" in parameter_name)
                        and not "whiten" in parameter_name
                    ):
                        incoming_net_parameter.copy_(ema_net_parameter.detach())

    def forward(self, inputs):
        with torch.no_grad():
            return self.net_ema(inputs)
