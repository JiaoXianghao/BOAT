import pytest

import bohml
import torch
from torch import nn
import numpy as np
from test_script.utils import ll_o, ul_o, ul_o_iapttgm

torch.manual_seed(123)

GNAME = "system-test-feature"


class TestSystemFeature:

    @pytest.mark.parametrize(
        "method, ll_method, ul_method, ll_objective, ul_objective, truncate_iter,"
        "truncate_max_loss_iter, alpha_init, alpha_decay, update_ll_model_init",
        [
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, True, 0.0, 0.0, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, False, 0.0, 0.0, True),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, True, 0.0, 0.0, True),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 1, False, 0.0, 0.0, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 1, False, 0.1, 0.5, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 1, False, 0.0, 0.0, True),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, False, 0.1, 0.5, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, True, 0.1, 0.5, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, False, 0.1, 0.5, True),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, True, 0.1, 0.5, True),
            ("Feature", "Dynamic", "Onestage", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "Dynamic", "Onestage", ll_o, ul_o, 0, False, 0.0, 0.0, True),
            ("Feature", "Implicit", "LS", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "Implicit", "NS", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "Implicit", "GN", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "BVFIM", "BVFIM", ll_o, ul_o, 0, False, 0.0, 0.0, False)
        ],
    )
    def test_few_shot_conv(self, method, ll_method, ul_method, ll_objective, ul_objective,
                      truncate_iter, truncate_max_loss_iter, alpha_init, alpha_decay,
                      update_ll_model_init):
        ul_model = bohml.utils.model.backbone.Conv([5, 3, 84, 84], 5, num_filters=32, use_head=False)

        ll_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(ul_model.output_shape[1:]), 5)
        )
        ll_opt = torch.optim.SGD(ll_model.parameters(), lr=0.1)
        ul_opt = torch.optim.Adam(ul_model.parameters(), eps=1e-5)
        tr_xs = torch.rand(4, 5, 3, 84, 84)
        tr_ys = torch.tensor([0, 1, 2, 3, 4]).repeat(4, 1)
        tst_xs = torch.rand(4, 75, 3, 84, 84)
        tst_ys = torch.tensor([0, 1, 2, 3, 4]).repeat(4, 15)

        optimizer = bohml.optimizer.BOHMLOptimizer(
            method, ll_method=ll_method, ul_method=ul_method,
            ll_objective=ll_o, ul_objective=ul_o_iapttgm if truncate_max_loss_iter else ul_o,
            ll_model=ll_model, ul_model=ul_model)
        optimizer.build_ll_solver(1 if ul_method == 'Onestage' else 5, ll_opt, truncate_iter=truncate_iter,
                                  truncate_max_loss_iter=truncate_max_loss_iter, alpha_init=alpha_init,
                                  alpha_decay=alpha_decay)
        optimizer.build_ul_solver(ul_opt, update_ll_model_init)

        for iter in range(5):
            val_loss, forward_time, backward_time = optimizer.run_iter(tr_xs, tr_ys, tst_xs, tst_ys,
                                                                       iter, forward_with_whole_batch=False)


    @pytest.mark.parametrize(
        "method, ll_method, ul_method, ll_objective, ul_objective, truncate_iter,"
        "truncate_max_loss_iter, alpha_init, alpha_decay, update_ll_model_init",
        [
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, True, 0.0, 0.0, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, False, 0.0, 0.0, True),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, True, 0.0, 0.0, True),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 1, False, 0.0, 0.0, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 1, False, 0.0, 0.0, True),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, False, 0.1, 0.5, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, True, 0.1, 0.5, False),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, False, 0.1, 0.5, True),
            ("Feature", "Dynamic", "Recurrence", ll_o, ul_o, 0, True, 0.1, 0.5, True),
            ("Feature", "Dynamic", "Onestage", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "Dynamic", "Onestage", ll_o, ul_o, 0, False, 0.0, 0.0, True),
            ("Feature", "Implicit", "LS", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "Implicit", "NS", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "Implicit", "GN", ll_o, ul_o, 0, False, 0.0, 0.0, False),
            ("Feature", "BVFIM", "BVFIM", ll_o, ul_o, 0, False, 0.0, 0.0, False)
        ],
    )
    def test_regular(self, method, ll_method, ul_method, ll_objective, ul_objective,
                     truncate_iter, truncate_max_loss_iter, alpha_init, alpha_decay,
                     update_ll_model_init):
        ul_model = bohml.utils.model.backbone.Conv([32, 3, 84, 84], 8, num_filters=32, use_head=False)

        ll_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(ul_model.output_shape[1:]), 10)
        )
        ll_opt = torch.optim.SGD(ll_model.parameters(), lr=0.1)
        ul_opt = torch.optim.Adam(ul_model.parameters(), eps=1e-5)
        tr_xs = torch.rand(32, 3, 84, 84)
        tr_ys = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(4)
        tst_xs = torch.rand(64, 3, 84, 84)
        tst_ys = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(8)

        optimizer = bohml.optimizer.BOHMLOptimizer(
            method, ll_method=ll_method, ul_method=ul_method,
            ll_objective=ll_o, ul_objective=ul_o_iapttgm if truncate_max_loss_iter else ul_o,
            ll_model=ll_model, ul_model=ul_model)
        optimizer.build_ll_solver(1 if ul_method == 'Onestage' else 5, ll_opt)
        optimizer.build_ul_solver(ul_opt)

        for iter in range(5):
            val_loss, forward_time, backward_time = optimizer.run_iter(tr_xs, tr_ys, tst_xs, tst_ys,
                                                                       iter, forward_with_whole_batch=True)
