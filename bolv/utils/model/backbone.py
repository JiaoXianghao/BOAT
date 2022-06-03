import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List


class Conv(nn.Module):
    def __init__(
            self,
            input_shape,
            ways,
            num_stages=4,
            num_filters=64,
            kernel_size=3,
            stride=1,
            padding=1,
            use_max_pool=True,
            max_padding=0,
            use_batch_norm=True,
            use_head=True,
            activation=nn.ReLU
    ):
        super(Conv, self).__init__()
        self._input_shape = input_shape
        self._output_shape = None
        self._num_classes = ways
        self._num_stages = num_stages
        self._num_filters = num_filters

        self._use_head = use_head

        self.create_model(input_shape[1], kernel_size, stride, padding, use_batch_norm,
                          use_max_pool, max_padding, use_head, activation)

    def create_model(
            self,
            input_channels,
            kernel_size,
            stride,
            padding,
            batch_norm,
            max_pool,
            max_padding,
            head,
            activation):

        nin = input_channels

        x = torch.zeros(self._input_shape)

        for idx in range(1, self._num_stages+1):
            conv = ConvBlock(input_channels=nin, output_channels=self._num_filters, kernel_size=kernel_size,
                             stride=stride, padding=padding, use_batch_norm=batch_norm, use_max_pool=max_pool,
                             max_padding=max_padding, activation=activation)
            setattr(self, 'block{}'.format(idx), conv)
            x = conv(x)
            nin = self._num_filters

        self._output_shape = x.shape

        if head:
            features = x.shape[2]
            head = nn.Linear(self._num_filters * features * features, self._num_classes)
            setattr(self, 'head', head)
        else:
            setattr(self, 'head', None)

    def forward(self, x):
        """Forward-pass through adapt_model."""
        for idx in range(1, self._num_stages+1):
            x = getattr(self, 'block{}'.format(idx))(x)

        if self._use_head:
            flatten = nn.Flatten()
            x = flatten(x)
            x = self.head(x)

        return x

    @property
    def num_filters(self):
        return self._num_filters

    @property
    def num_stages(self):
        return self._num_stages

    @property
    def output_shape(self):
        return self._output_shape


class Res12(nn.Module):
    def __init__(
            self,
            input_shape: List[int],
            ways: int,
            use_head: bool = True
    ):
        super(Res12, self).__init__()
        self._input_shape = input_shape
        self._output_shape = None
        self._num_classes = ways
        self._num_stages = 4

        self.create_model(use_head)

    def create_model(self, use_head):
        nin = self._input_shape[1]
        x = torch.zeros(self._input_shape)

        num_chn = [64, 128, 256, 512]
        max_padding = [0, 0, 1, 1]
        maxpool = [True, True, True, False]
        for idx in range(len(num_chn)):
            res = ResBlock(input_channels=nin, num_filters=num_chn[idx], max_pool=maxpool[idx],
                           max_padding=max_padding[idx], normalization=True, use_bias=False)
            setattr(self, 'block{}'.format(idx+1), res)
            x = res(x)
            nin = num_chn[idx]

        x = F.adaptive_avg_pool2d(x, (1, 1))

        self._output_shape = x.shape

        if use_head:
            linear = nn.Linear(np.prod(x.shape[1:]), self._num_classes)
            setattr(self, 'head', linear)
        else:
            setattr(self, 'head', None)

    def forward(self, x):
        """Forward-pass through adapt_model."""
        for idx in range(1, self._num_stages+1):
            x = getattr(self, 'block{}'.format(idx))(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))

        if self.head is not None:
            flatten = nn.Flatten()
            x = flatten(x)
            x = self.head(x)

        return x

    @property
    def output_shape(self):
        return self._output_shape


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, use_bias=True,
                 use_batch_norm=True, use_max_pool=True, max_padding=0, use_activation=True,
                 activation=nn.ReLU):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param normalization: The type of normalization to use 'use_batch_norm' or 'layer_norm'
           meta-conv etc.
           :param input_channels: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param use_max_pool: whether use carpool in this layer or not
           :param max_padding: if use_max_pool is True, the number of paddings for max-pool
           :param normalization: whether use batch-norm in this layer
        """
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, (kernel_size, kernel_size), padding=padding,
                              stride=(stride, stride), bias=use_bias)
        self.activation = activation(inplace=True) if use_activation else None
        self.norm = nn.BatchNorm2d(output_channels, momentum=1., affine=True,
                                   track_running_stats=False) if use_batch_norm else None
        self.max_pool = nn.MaxPool2d((2, 2), 2, max_padding) if use_max_pool else None

    def forward(self, x, params=None):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param x: input data batch, size either can be any.
            :param params: parameters of current block list.
            :return: The result of the batch norm operation.
        """
        if params is None:
            x = self.conv(x)
            x = self.activation(x) if self.activation is not None else x
            x = self.max_pool(x) if self.max_pool is not None else x
            x = self.norm(x) if self.norm is not None else x
        else:
            x = F.conv2d(x, next(params), next(params), stride=1, padding=1)
            x = F.relu(x, inplace=True) if self.activation is not None else x
            x = F.max_pool2d(x, 2) if self.max_pool is not None else x
            x = F.batch_norm(x, torch.zeros(np.prod(np.array(x.data.size()[1]))).cuda(),
                             torch.ones(np.prod(np.array(x.data.size()[1]))).cuda(),
                             next(params), next(params), training=True, momentum=1)\
                if self.norm is not None else x

        return x


class ResBlock(nn.Module):
    def __init__(self, input_channels, num_filters, max_pool, max_padding, normalization=True, use_bias=False):
        """

        Parameters
        ----------
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param normalization: The type of normalization to use 'use_batch_norm' or 'layer_norm'
           meta-conv etc.
           :param input_channels: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param max_pool: whether use carpool in this layer or not
           :param max_padding: if use_max_pool is True, the number of paddings for max-pool
           :param normalization: whether use batch-norm in this layer
        """
        super(ResBlock, self).__init__()

        self.conv1 = ConvBlock(input_channels, num_filters, kernel_size=3, padding=1, stride=1, use_bias=use_bias,
                               use_max_pool=False, use_batch_norm=normalization)
        self.conv2 = ConvBlock(num_filters, num_filters, kernel_size=3, padding=1, stride=1, use_bias=use_bias,
                               use_max_pool=False, use_batch_norm=normalization)
        self.conv3 = ConvBlock(num_filters, num_filters, kernel_size=3, padding=1, stride=1, use_bias=use_bias,
                               use_max_pool=False, use_activation=False)
        self.conv_blocks = nn.Sequential(*[self.conv1, self.conv2, self.conv3])

        self.shortcut_conv = ConvBlock(input_channels, num_filters, kernel_size=1, stride=1, padding=0,
                                       use_bias=use_bias, use_max_pool=False, use_activation=False)

        self.max_pool = nn.MaxPool2d((2, 2), 2, max_padding) if max_pool else None

    def forward(self, x):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param x: input data batch, size either can be any.
            :return: The result of the batch norm operation.
        """

        out = x
        identity = x

        out = self.conv_blocks(out)
        identity = self.shortcut_conv(identity)

        out += identity
        out = F.relu(out)

        if self.max_pool is not None:
            out = self.max_pool(out)

        return out
