import torch
from .ul_optimizer import ULGrad
from torch.autograd import grad as torch_grad

from torch.nn import Module
from torch import Tensor
from typing import List, Callable
from higher.patch import _MonkeyPatchBase

from bohml.utils.utils import update_grads


class NS(ULGrad):
    r"""Calculation of the gradient of the upper adapt_model variables with Implicit Gradient Based Methods.

    Implements the UL problem optimization procedure of two implicit gradient
    based methods (IGBMs), neumann series based method (NS) `[1]`_.

    A wrapper of lower adapt_model that has been optimized in the LL optimization will
    be used in this procedure.

    Parameters
    ----------
        ul_objective: callable
            The main optimization problem in a hierarchical optimization problem.

            Callable with signature callable(state). Defined based on modeling of
            the specific problem that need to be solved. Computing the loss of upper
            problem. The state object contains the following:

            - "data"
                Data used in the upper optimization phase.
            - "target"
                Target used in the upper optimization phase.
            - "ul_model"
                Upper adapt_model of the bi-level adapt_model structure.
            - "ll_model"
                Lower adapt_model of the bi-level adapt_model structure.

        ul_model: Module
            Upper adapt_model in a hierarchical adapt_model structure whose parameters will be
            updated with upper objective and trained lower adapt_model.

        ll_objective: callable
            An optimization problem which is considered as the constraint of upper
            level problem.

            Callable with signature callable(state). Defined based on modeling of
            the specific problem that need to be solved. Computing the loss of upper
            problem. The state object contains the following:

            - "data"
                Data used in the upper optimization phase.
            - "target"
                Target used in the upper optimization phase.
            - "ul_model"
                Upper adapt_model of the bi-level adapt_model structure.
            - "ll_model"
                Lower adapt_model of the bi-level adapt_model structure.

        ll_model: Module
            Lower adapt_model in a hierarchical adapt_model structure whose parameters will be
            updated with lower objective during lower-level optimization.

        lower_learning_rate: float
            Step size for lower loop optimization.

        k: int
            The maximum number of conjugate gradient iterations.

        tolerance: float, default=1e-10
            End the method earlier when the norm of the residual is less than tolerance.

    References
    ----------
    _`[1]`  J. Lorraine, P. Vicol, and D. Duvenaud, "Optimizing millions of
     hyperparameters by implicit differentiation", in AISTATS, 2020.
    """

    def __init__(
            self,
            ul_objective: Callable[[Tensor, Tensor, Module, Module], Tensor],
            ul_model: Module,
            ll_objective: Callable[[Tensor, Tensor, Module, Module], Tensor],
            ll_model: Module,
            lower_learning_rate: float,
            k: int,
            tolerance: float = 1e-10
    ):
        super(NS, self).__init__(ul_objective, ul_model, ll_model)

        self.K = k
        self.lower_learning_rate = lower_learning_rate
        self.ll_objective = ll_objective
        self.tolerance = tolerance

    def compute_gradients(
            self,
            validate_data: Tensor,
            validate_target: Tensor,
            auxiliary_model: _MonkeyPatchBase,
            train_data: Tensor,
            train_target: Tensor
    ):
        """
        Compute the grads of upper variable with validation data samples in the batch
        using upper objective. The grads will be saved in the passed in upper adapt_model.

        Note that the implemented UL optimization procedure will only compute
        the grads of upper variables。 If the validation data passed in is only single data
        of the batch (such as few-shot learning experiment), then compute_gradients()
        function should be called repeatedly to accumulate the grads of upper variables
        for the whole batch. After that the update operation of upper variables needs
        to be done outside this module.

        Parameters
        ----------
            validate_data: Tensor
                The validation data used for UL problem optimization.

            validate_target: Tensor
                The labels of the samples in the validation data.

            auxiliary_model: _MonkeyPatchBase
                Wrapper of lower adapt_model encapsulated by module higher, has been optimized in LL
                optimization phase.

            train_data: Tensor
                The training data used for LL problem optimization.

            train_target: Tensor
                The labels of the samples in the train data.

        Returns
        -------
        upper_loss: Tensor
            Returns the loss value of upper objective.
        """

        hparams = list(self.ul_model.parameters())

        def fp_map(params, loss_f):
            lower_grads = list(torch.autograd.grad(loss_f, params, create_graph=True))
            updated_params = []
            for i in range(len(params)):
                updated_params.append(params[i] - self.lower_learning_rate * lower_grads[i])
            return updated_params

        lower_model_params = list(auxiliary_model.parameters())

        lower_loss = self.ll_objective(train_data, train_target, self.ul_model, auxiliary_model)
        upper_loss = self.ul_objective(validate_data, validate_target, self.ul_model, auxiliary_model)

        upper_grads = neumann(lower_model_params, hparams, upper_loss, lower_loss, self.K, fp_map, self.tolerance)

        update_grads(upper_grads, self.ul_model)

        return upper_loss


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


def neumann(params: List[Tensor],
            hparams: List[Tensor],
            upper_loss,
            lower_loss,
            k: int,
            fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
            tol=1e-10) -> List[Tensor]:
    """ Saves one iteration from the fixed point method"""

    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)

    w_mapped = fp_map(params, lower_loss)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for i in range(k):
        gs_prev_vec = gs_vec
        vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(torch.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = torch_grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    return grads


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, list(inputs)))
