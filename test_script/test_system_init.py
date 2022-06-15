import pytest

import bolv
import torch
from test_script.utils import inner_o, outer_o

torch.manual_seed(123)

GNAME = "system-test-init"


class TestSystemFeature:

    @pytest.mark.parametrize(
        "learn_lr, use_t, use_warp, use_forget",
        [
            (False, False, False, False),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (False, False, False, True),
            (True, True, False, False),
            (True, False, True, False),
            (True, False, False, True),
        ],
    )
    def test_few_shot(self, learn_lr, use_t, use_warp, use_forget):
        backbone_model = bolv.utils.model.backbone.Conv([5, 3, 84, 84], 5, num_filters=32, use_head=True)
        meta_model = bolv.utils.model.MetaModel(backbone_model, learn_lr=learn_lr, use_t=use_t,
                                                use_warp=use_warp, use_forget=use_forget)
        outer_opt = torch.optim.Adam(meta_model.parameters())

        optimizer = bolv.optimizer.BOLVOptimizer("Initial", inner_objective=inner_o,
                                                 outer_objective=outer_o, meta_model=meta_model)
        optimizer.build_meta_solver(outer_opt, 3, inner_learning_rate=0.01, learn_lr=learn_lr, use_t=use_t,
                                    use_warp=use_warp, use_forget=use_forget)

        tr_xs = torch.rand(4, 5, 3, 84, 84)
        tr_ys = torch.tensor([0, 1, 2, 3, 4]).repeat(4, 1)
        tst_xs = torch.rand(4, 75, 3, 84, 84)
        tst_ys = torch.tensor([0, 1, 2, 3, 4]).repeat(4, 15)

        for iter in range(5):
            val_loss, forward_time, backward_time = optimizer.run_iter(tr_xs, tr_ys, tst_xs, tst_ys,
                                                                       iter, forward_with_whole_batch=False)

    @pytest.mark.parametrize(
        "learn_lr, use_t, use_warp, use_forget",
        [
            (False, False, False, False),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (False, False, False, True),
            (True, True, False, False),
            (True, False, True, False),
            (True, False, False, True),
        ],
    )
    def test_regular(self, learn_lr, use_t, use_warp, use_forget):
        backbone_model = bolv.utils.model.backbone.Conv([32, 3, 84, 84], 8, num_filters=32, use_head=True)
        meta_model = bolv.utils.model.MetaModel(backbone_model, learn_lr=learn_lr, use_t=use_t,
                                                use_warp=use_warp, use_forget=use_forget)
        outer_opt = torch.optim.Adam(meta_model.parameters())

        optimizer = bolv.optimizer.BOLVOptimizer("Initial", inner_objective=inner_o,
                                                 outer_objective=outer_o, meta_model=meta_model)
        optimizer.build_meta_solver(outer_opt, 3, inner_learning_rate=0.01, learn_lr=learn_lr, use_t=use_t,
                                    use_warp=use_warp, use_forget=use_forget)

        tr_xs = torch.rand(32, 3, 84, 84)
        tr_ys = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(4)
        tst_xs = torch.rand(64, 3, 84, 84)
        tst_ys = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(8)

        for iter in range(5):
            val_loss, forward_time, backward_time = optimizer.run_iter(tr_xs, tr_ys, tst_xs, tst_ys,
                                                                       iter, forward_with_whole_batch=True)
