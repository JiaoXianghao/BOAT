import torch

from torch.nn import functional as F
import numpy as np
from torch import nn

from ..utils.utils import update_grads

from ..utils.model.meta_model import MetaModel
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Callable, List, Dict


# def _get_params_dict(model):
#     params_dict = {}
#     for idx in range(1, model.num_stages+1):
#         params_dict['block{}'.format(idx)] = list(getattr(model, 'block{}'.format(idx)).parameters())
#     return params_dict
def rearrange_named_params(named_params):
    new_dict = {'name': [], 'params': []}
    for name, param in named_params:
        new_dict['name'].append(name)
        new_dict['params'].append(param)
    return new_dict


class Init(object):
    r"""Complete Meta-learning Process with MAML and MAML-based methods

    Implements the meta learning procedure of MAML_`[1]`_ and four MAML based methods,
    Meta-SGD_`[2]`_, MT-net_`[3]`_, Warp-grad_`[4]`_ and L2F_`[5]`_.

    Parameters
    ----------
        model: MetaModel
            Model wrapped by class MetaModel which contains backbone network and other auxiliary meta modules if using
            other MAML-based methods.

        inner_objective: callable
            The inner loop optimization objective.

            Callable with signature callable(state). Defined based on modeling of
            the specific problem that need to be solved. Computing the loss of inner
            objective. The state object contains the following:

            - "data"
                Data used in inner optimization phase.
            - "target"
                Target used in inner optimization phase.
            - "adapt_model"
                Meta adapt_model to be updated.
            - "updated_weights"
                Weights of adapt_model updated in inner-loop, will be used for forward propagation.


        outer_objective: callable
            The outer optimization objective.

            Callable with signature callable(state). Defined based on modeling of
            the specific problem that need to be solved. Computing the loss of outer
            objective. The state object contains the following:

            - "data"
                Data used in outer optimization phase.
            - "target"
                Target used in outer optimization phase.
            - "adapt_model"
                Meta adapt_model to be updated.
            - "updated_weights"
                Weights of adapt_model updated in inner-loop, will be used for forward propagation.

        inner_learning_rate: float, default=0.01
            Step size for inner optimization.

        inner_loop (optional): int, default=5
            Num of inner optimization steps.

        use_second_order (optional): bool, default=True
            Optional argument,whether to calculate precise second-order gradients during inner-loop.

        learn_lr (optional): bool, default=False
            Optional argument, whether to update inner learning rate during outer optimization,
            i.e. use MSGD method.

        use_t (optional): bool, default=False
            Optional argument, whether to using T-layers during optimization,i.e. use MT-net method.

        use_warp (optional): bool, default=False
            Optional argument, whether to using warp modules during optimization,i.e. use Warp-grad method.

        use_forget (optional): bool, default=False
            Optional argument, whether to add attenuation to each layers, i.e. use L2F method.

    References
    ----------
    _`[1]` C. Finn, P. Abbeel, S. Levine, "Model-Agnostic Meta-Learning for
    Fast Adaptation of Deep Networks", in ICML, 2017.

    _`[2]` Z. Li, F. Zhou, F. Chen, H. Li, "Meta-SGD: Learning to Learn Quickly for
    Few-Shot Learning", in arxiv, 2017.

    _`[3]` Y. Lee and S. Choi, "Gradient-Based Meta-Learning with Learned Layer-wise
    Metric and Subspace", in ICML, 2018.

    _`[4]` S. Flennerhag, A. Rusu, R. Pascanu, F. Visin, H. Yin, R. Hadsell, "Meta-learning
    with Warped Gradient Descent", in ICLR, 2020.

    _`[5]` S. Baik, S. Hong, K. Lee, "Learning to Forget for Meta-Learning", in CVPR, 2020.
    """

    def __init__(
            self,
            model: MetaModel,
            inner_objective: Callable[[Tensor, Tensor, Module, Dict[str, List[Parameter]]], Tensor],
            outer_objective: Callable[[Tensor, Tensor, Module, Dict[str, List[Parameter]]], Tensor],
            inner_learning_rate: float = 0.01,
            inner_loop: int = 5,
            use_second_order: bool = True,
            learn_lr: bool = False,
            use_t: bool = False,
            use_warp: bool = False,
            use_forget: bool = False
    ):
        super(Init, self).__init__()
        self._model = model
        self._inner_objective = inner_objective
        self._outer_objective = outer_objective

        self._learn_lr = learn_lr
        self._use_t = use_t
        self._use_warp = use_warp
        self._use_forget = use_forget

        self._inner_loop = inner_loop
        self._inner_learning_rate = inner_learning_rate
        self._use_second_order = use_second_order

    def optimize(
            self,
            train_data: Tensor,
            train_target: Tensor,
            validate_data: Tensor,
            validate_target: Tensor
    ):
        """
        The meta optimization process containing bolv inner loop phase and outer loop phase.
         Final grads will be calculated by outer objective and saved in the passed in adapt_model.

        Note that the implemented optimization procedure will compute the grads of meta adapt_model
        with only one single set of training and validation data samples in a batch. If
        batch size is larger than 1, then optimize() function should be called repeatedly to
        accumulate the grads of adapt_model variables for the whole batch. After that the update
        operation of adapt_model variable needs to be done outside this optimization module.

        Parameters
        ----------
            train_data: Tensor
                The training data used in inner loop phase.

            train_target: Tensor
                The labels of the samples in the train data.

            validate_data: Tensor
                The validation data used in outer loop phase.

            validate_target: Tensor
                The labels of the samples in the validation data.

        Returns
        -------
        val_loss: Tensor
            Value of validation loss.
        """
        named_adapt_params_dict = rearrange_named_params(self._model.adapt_model.named_parameters())

        if self._use_forget:
            loss = self._inner_objective(train_data, train_target, self._model, named_adapt_params_dict)
            attenuated_params = self._model.get_attenuated_params(loss)  # todo
            named_adapt_params_dict = rearrange_named_params(attenuated_params)

        # inner optimization
        # params = list(self._model.parameters())
        for y_idx in range(self._inner_loop):

            #     out = forward(train_data, params)
            #     loss = F.cross_entropy(out, train_target)
            #     adapt_grads = torch.autograd.grad(loss, params, retain_graph=True,
            #                                       create_graph=self._use_second_order)
            #     for idx in range(len(params)):
            #         params[idx] = params[idx] - self._inner_learning_rate * adapt_grads[idx]
            inner_loss = self._inner_objective(train_data, train_target, self._model, named_adapt_params_dict)
            adapt_grads = torch.autograd.grad(inner_loss, named_adapt_params_dict['params'], retain_graph=True,
                                              create_graph=self._use_second_order)
            for idx in range(len(named_adapt_params_dict['params'])):
                if self._learn_lr:
                    named_adapt_params_dict['params'][idx] = named_adapt_params_dict['params'][idx] - \
                                                             self._model.lr * adapt_grads[idx]
                else:
                    named_adapt_params_dict['params'][idx] = named_adapt_params_dict['params'][idx] - \
                                                             self._inner_learning_rate * adapt_grads[idx]
            # print(named_adapt_params_dict['params'][16])

        # out = forward(validate_data, params)
        # outer_loss = F.cross_entropy(out, validate_target)

        # outer optimization
        outer_loss = self._outer_objective(validate_data, validate_target, self._model, named_adapt_params_dict)
        # meta_grads = torch.autograd.grad(outer_loss, list(self._model.parameters()))
        # print(meta_grads)
        # update_grads(meta_grads, self._model)
        outer_loss.backward()

        return outer_loss


def forward(x, params):
    for i in range(1, 5):
        x = F.conv2d(x, params[4 * (i - 1)], params[4 * (i - 1) + 1], stride=1, padding=1)

        x = F.relu(x, inplace=True)
        x = F.batch_norm(x, torch.zeros(np.prod(np.array(x.data.size()[1]))).cuda(),
                         torch.ones(np.prod(np.array(x.data.size()[1]))).cuda(),
                         params[4 * (i - 1) + 2], params[4 * (i - 1) + 3], training=True, momentum=1)
        x = F.max_pool2d(x, 2)

    flatten = nn.Flatten()
    x = flatten(x)
    x = F.linear(x, params[16], params[17])
    return x


def acc(out, target):
    pred = out.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item() / len(target)
