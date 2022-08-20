import pytest

import torch
from torch import nn

import bohml.optimizer

from test_script.utils import ll_o, ul_o, inner_o, outer_o

GNAME = "errors-warnings"


class TestErrorsWarnings:

    @pytest.mark.parametrize(
        "method, ll_method, ul_method, ll_objective, ul_objective, inner_objective, outer_objective",
        [
            ("Feat", None, None, None, None, None, None),
            ("Feature", None, None, None, None, None, None),
            ("Feature", "Dynamic", None, None, None, None, None),
            ("Feature", "Dynamic", "Recurrence", None, None, None, None),
            ("Feature", "Dynamic", "Recurrence", ll_o, None, None, None),
            ("Feature", "Dynamic", "Recurrence", None, ul_o, None, None),
            ("Feature", "Recurrence", None, None, None, None, None),
            ("Feature", "Dynamic", "Dynamic", None, None, None, None),
            ("Feature", "Dynamic", "LS", None, None, None, None),
            ("Feature", "Dynamic", "NS", None, None, None, None),
            ("Feature", "Dynamic", "GN", None, None, None, None),
            ("Feature", "Dynamic", "BVFIM", None, None, None, None),
            ("Feature", "Implicit", "Recurrence", None, None, None, None),
            ("Feature", "Implicit", "Onestage", None, None, None, None),
            ("Feature", "Implicit", "BVFIM", None, None, None, None),
            ("Feature", "BVFIM", "Recurrence", None, None, None, None),
            ("Feature", "BVFIM", "Onestage", None, None, None, None),
            ("Feature", "BVFIM", "LS", None, None, None, None),
            ("Feature", "BVFIM", "NS", None, None, None, None),
            ("Feature", "BVFIM", "GN", None, None, None, None),
            ("Initial", None, None, None, None, None, None),
            ("Initial", None, None, None, None, None, outer_o),
            ("Initial", None, None, None, None, inner_o, None),
            ("Initial", None, None, None, None, inner_o, outer_o),
        ],
    )
    def test_bolvoptimizer_init_method(self, method, ll_method, ul_method, ll_objective, ul_objective,
                                       inner_objective, outer_objective):
        with pytest.raises(AssertionError):
            ul_model = nn.Conv2d(3, 3, (3, 3))
            ll_model = nn.Conv2d(3, 3, (3, 3))
            bohml.optimizer.BOHMLOptimizer(method, ll_method, ul_method, ll_objective, ul_objective,
                                           inner_objective, outer_objective, ll_model, ul_model, None)

    def test_bolvoptimizer_init_model_1(self):
        with pytest.raises(AssertionError):
            ll_model = nn.Conv2d(3, 3, (3, 3))
            bohml.optimizer.BOHMLOptimizer("Feature", "Dynamic", "Recurrence", ll_o, ul_o,
                                           None, None, ll_model, None, None)

    def test_bolvoptimizer_init_model_2(self):
        with pytest.raises(AssertionError):
            ul_model = nn.Conv2d(3, 3, (3, 3))
            bohml.optimizer.BOHMLOptimizer("Feature", "Dynamic", "Recurrence", ll_o, ul_o,
                                           None, None, None, ul_model, None)

    def test_bolvoptimizer_init_model_3(self):
        with pytest.raises(TypeError):
            meta_model = nn.Conv2d(3, 3, (3, 3))
            bohml.optimizer.BOHMLOptimizer("Initial", None, None, None, None,
                                           inner_o, outer_o, None, None, meta_model)

    @pytest.mark.parametrize(
        "ul_method, lower_loop, truncate_iter, "
        "truncate_max_loss_iter, alpha_init, alpha_decay",
        [
            ("Onestage", 5, 0, False, 0.0, 0.0),
            ("Onestage", 1, 1, False, 0.0, 0.0),
            ("Onestage", 1, 0, True, 0.0, 0.0),
            ("Recurrence", 5, 1, True, 0.0, 0.0),
            ("Recurrence", 5, 0, True, 0.5, 0.0),
            ("Recurrence", 5, 0, False, 1.1, 0.0),
            ("Recurrence", 5, 0, False, 0.4, 1.1),
            ("Recurrence", 5, 0, False, 0.4, 0.0),
            ("Recurrence", 5, 6, False, 0.0, 0.0)
        ],
    )
    def test_bothoptimizer_build_ll_solver(self, ul_method, lower_loop, truncate_iter,
                                           truncate_max_loss_iter, alpha_init, alpha_decay):
        with pytest.raises(AssertionError):
            ll_model = nn.Conv2d(3, 3, (3, 3))
            ul_model = nn.Conv2d(3, 3, (3, 3))
            optimizer = bohml.optimizer.BOHMLOptimizer("Feature", "Dynamic", ul_method, ll_o, ul_o,
                                                       None, None, ll_model, ul_model, None)
            ll_opt = torch.optim.SGD(ll_model.parameters(), lr=0.01)
            optimizer.build_ll_solver(lower_loop, ll_opt, truncate_iter, truncate_max_loss_iter,
                                      alpha_init, alpha_decay)

    @pytest.mark.parametrize(
        "ll_method, ul_method",
        [
            ("Implicit", "GN"),
            ("BVFIM", "BVFIM")
        ]
    )
    def test_bothoptimizer_build_ul_solver(self, ll_method, ul_method):
        with pytest.raises(AssertionError):
            ll_model = nn.Conv2d(3, 3, (3, 3))
            ul_model = nn.Conv2d(3, 3, (3, 3))
            optimizer = bohml.optimizer.BOHMLOptimizer("Feature", ll_method, ul_method, ll_o, ul_o,
                                                       None, None, ll_model, ul_model, None)
            ll_opt = torch.optim.SGD(ll_model.parameters(), lr=0.01)
            optimizer.build_ll_solver(5, ll_opt)
            ul_opt = torch.optim.SGD(ul_model.parameters(), lr=0.001)
            optimizer.build_ul_solver(ul_opt, update_ll_model_init=True)

