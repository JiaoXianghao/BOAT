import pytest

import torch
from torch import nn
import numpy as np
import bohml
from bohml.utils.model.backbone import Conv


backbone_model = Conv([5, 3, 84, 84], 5, num_filters=32, use_head=True)

tr_xs = torch.rand(32, 3, 84, 84)
tr_ys = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(4)
tst_xs = torch.rand(64, 3, 84, 84)
tst_ys = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(8)


class TestBackbone:

    def test_bilevel_model_conv4(self):
        ul_model = bohml.utils.model.backbone.Conv([32, 3, 84, 84], 8, num_filters=32, use_head=False)

        ll_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(ul_model.output_shape[1:]), 10)
        )
        ll_model(ul_model(tr_xs))

    def test_bilevel_model_res12(self):
        ul_model = bohml.utils.model.backbone.Res12([32, 3, 84, 84], 8, use_head=False)

        ll_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(ul_model.output_shape[1:]), 10)
        )
        ll_model(ul_model(tr_xs))

    def test_singlelevel_model_conv4(self):
        model = bohml.utils.model.backbone.Conv([32, 3, 84, 84], 8, num_filters=32, use_head=True)
        model(tr_xs)

    def test_singlelevel_model_res12(self):
        model = bohml.utils.model.backbone.Res12([32, 3, 84, 84], 8, use_head=True)
        model(tr_xs)
