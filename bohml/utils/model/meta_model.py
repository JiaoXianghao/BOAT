import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Module
from .backbone import ConvBlock
from typing import Dict


def extract_block_params(block_name, named_params_dict):
    for idx in range(len(named_params_dict['name'])):
        if block_name in named_params_dict['name'][idx]:
            yield named_params_dict['params'][idx]


class MetaModel(nn.Module):
    r"""
    Special  adapt_model used for initialization optimization with MAML and MAML based methods.
    Containing backbone adapt_model(CONV4, for example) and additional modules.

    Parameters
    ----------
        backbone: Module
            Backbone adapt_model, could

        learn_lr: bool, default=False
            Whether to learning inner learning rate during outer optimization,
            i.e. use Meta-SGD method.

        meta_lr: float, default=0.1
            Learning rate of inner optimization.

        use_t: bool, default=False
            Whether to add T-layers, i.e. use MT-net method.

        use_warp: bool, default=False
            Whether to add Warp-blocks, i.e. use Warp-grad method.

        num_warp_layers: int, default=1
            Num of conv layers in one warp block.

        use_forget: bool, default=False
            Whether to add attenuator, i.e. use Learning-to-Forget method.

        enable_inner_loop_optimizable_bn_params: bool, default=False
            When use L2F method, whether to add the attenuation operation to the batch-norm modules.
    """
    def __init__(
            self,
            backbone: Module,
            learn_lr: bool = False,
            meta_lr: float = 0.1,
            use_t: bool = False,
            use_warp: bool = False,
            num_warp_layers: int = 1,
            use_forget: bool = False,
            enable_inner_loop_optimizable_bn_params: bool = False
    ):
        super(MetaModel, self).__init__()

        assert (not use_t and not use_warp and not use_forget) or \
               (use_t and not use_warp and not use_forget) or \
               (not use_t and use_warp and not use_forget) or \
               (not use_t and not use_warp and use_forget),\
               "Only one of method 'MT-net', 'Warp-grad' or 'L2F' could be chosen."

        self.adapt_model = backbone

        # MSGD method setting
        self.learn_lr = learn_lr
        self.lr = nn.Parameter(torch.tensor(meta_lr), requires_grad=True) if learn_lr else None

        # MT-net method setting
        self.use_t = use_t
        if use_t:
            for idx in range(1, self.adapt_model.num_stages + 1):
                setattr(self, 'T{}'.format(idx),
                        torch.nn.Parameter(torch.eye(self.adapt_model.num_filters,
                                                     self.adapt_model.num_filters).unsqueeze(dim=2).unsqueeze(
                            dim=3).requires_grad_(True)))

        # Warp-grad method setting
        self.use_warp = use_warp
        self.num_warp_layers = num_warp_layers
        # self.warp_final_head = warp_final_head
        if use_warp:
            for idx in range(1, self.adapt_model.num_stages + 1):
                for _ in range(1, num_warp_layers + 1):
                    setattr(self, 'warp{}{}'.format(idx, _),
                            ConvBlock(self.adapt_model.num_filters, self.adapt_model.num_filters,
                                      padding=1, use_activation=False,
                                      use_max_pool=False, use_batch_norm=False))

        # L2F method setting
        self.use_forget = use_forget
        self.attenuate_bn_params = enable_inner_loop_optimizable_bn_params
        self.attenuator = self.get_mlp_attenuator() if use_forget else None

    def forward(
            self,
            x,
            named_params: Dict = None
    ):
        """
        Forward propagates by applying the function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        """
        #
        # assert len(list(params)) == len(list(self.adapt_model.parameters())), \
        #     "Passed in parameters should match adapt modules"

        for idx in range(1, self.adapt_model.num_stages + 1):
            if named_params is not None:
                block_params = extract_block_params('block{}'.format(idx), named_params)
                x = getattr(self.adapt_model, 'block{}'.format(idx))(x, block_params)
            else:
                x = getattr(self.adapt_model, 'block{}'.format(idx))(x)

            if self.use_t:
                x = F.conv2d(x, getattr(self, 'T{}'.format(idx)), stride=1)
            elif self.use_warp:
                for j in range(1, self.num_warp_layers + 1):
                    x = getattr(self, 'warp{}{}'.format(idx, j))(x)

        if self.adapt_model.head is not None:
            flatten = nn.Flatten()
            x = flatten(x)
            if named_params is not None:
                head_params = named_params['params'][-2:]
                x = F.linear(x, head_params[0], head_params[1])
            else:
                x = self.adapt_model.head(x)

        return x

    def adapt_modules(self):
        """Iterator for task-adaptable modules"""
        return self.adapt_model.parameters()

    @property
    def t_modules(self):
        if self.use_t:
            for name, param in self.named_parameters():
                if 'T' in name:
                    yield param
        else:
            return None

    @property
    def warp_modules(self):
        """Iterator for warp-layer modules"""
        if self.use_warp:
            for name, param in self.named_parameters():
                if 'warp' in name:
                    yield param
        else:
            return None

    def get_mlp_attenuator(self):
        num_attenuate_layers = len(list(self.adapt_model.parameters())) \
            if self.attenuate_bn_params else len(list(self.extract_conv_params()))
        attenuator = nn.Sequential(
            nn.Linear(num_attenuate_layers, num_attenuate_layers),
            nn.ReLU(inplace=True),
            nn.Linear(num_attenuate_layers, num_attenuate_layers),
            nn.Sigmoid()
        )
        return attenuator

    def extract_conv_params(self):
        """"""
        for name, param in self.adapt_model.named_parameters():
            if param.requires_grad:
                if 'norm' not in name:
                    yield param

    def get_attenuated_params(self, loss):
        if self.attenuate_bn_params:
            grads = torch.autograd.grad(loss, list(self.adapt_model.parameters()), create_graph=False)
        else:
            grads = torch.autograd.grad(loss, list(self.extract_conv_params()), create_graph=False)
        layerwise_mean_grads = []
        for i in range(len(grads)):
            layerwise_mean_grads.append(grads[i].mean())
        layerwise_mean_grads = torch.stack(layerwise_mean_grads)
        gamma = self.attenuator(layerwise_mean_grads)

        gamma = iter(gamma)
        for name, param in self.adapt_model.named_parameters():
            if self.attenuate_bn_params:
                yield name, param * next(gamma)
            else:
                if 'norm' not in name:
                    yield name, param * next(gamma)
                else:
                    yield name, param
