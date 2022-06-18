import pytest

import bolv
from bolv.utils.model.backbone import Conv


backbone_model = Conv([5, 3, 84, 84], 5, num_filters=32, use_head=True)


class TestMetaModel:

    @pytest.mark.parametrize(
        "use_t, use_warp, use_forget ",
        [
            (True, True, True),
            (False, True, True),
            (True, False, True),
            (True, True, False)
        ],
    )
    def test_init(self, use_t, use_warp, use_forget):
        with pytest.raises(AssertionError):
            meta_model = bolv.utils.model.MetaModel(backbone_model, use_t=use_t,
                                                    use_warp=use_warp, use_forget=use_forget)

    def test_msgd_model(self):
        meta_model = bolv.utils.model.MetaModel(backbone_model, learn_lr=True, meta_lr=0.01)

        assert meta_model.lr is not None

    def test_tnet_model(self):
        meta_model = bolv.utils.model.MetaModel(backbone_model, use_t=True)

        assert meta_model.t_modules is not None

    def test_warpgrad_model(self):
        meta_model = bolv.utils.model.MetaModel(backbone_model, use_warp=True)

        assert meta_model.warp_modules is not None

    def test_l2f_model(self):
        meta_model = bolv.utils.model.MetaModel(backbone_model, use_forget=True)

        assert meta_model.attenuator is not None

    def test_tnet_with_msgd_model(self):
        meta_model = bolv.utils.model.MetaModel(backbone_model, learn_lr=True, use_t=True)

        assert meta_model.t_modules is not None and meta_model.lr is not None

    def test_warpgrad_with_msgd_model(self):
        meta_model = bolv.utils.model.MetaModel(backbone_model, learn_lr=True, use_warp=True)

        assert meta_model.warp_modules is not None and meta_model.lr is not None

    def test_l2f_with_msgd_model(self):
        meta_model = bolv.utils.model.MetaModel(backbone_model, learn_lr=True, use_forget=True)

        assert meta_model.attenuator is not None and meta_model.lr is not None

