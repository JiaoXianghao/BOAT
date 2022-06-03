import torch
from bolv.ul_optimize.ul_optimizer import UpperGrad
from ..utils.utils import update_grads

from torch.nn import Module
from torch import Tensor
from typing import Callable
from higher.patch import _MonkeyPatchBase


class Onestage(UpperGrad):
    r"""Calculation of the gradient of the upper adapt_model variables with DARTS method

    Implements the UL optimization procedure of DARTS _`[1]`_, a first order approximation
    method which is free of bolv second-order derivatives and matrix-vector products.

    A wrapper of lower adapt_model that has been optimized in the lower optimization will
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
            updated with upper objective.

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

        update_ll_model_init: bool, default=False
            If set True, the initial value of ll model will be updated after this iteration.

        r: float, default=1e-2
           Parameter to adjust scalar $\epsilon$ as: $\epsilon = 0.01/\|
           \nabla_{w'}\mathcal L_{val}(w',\alpha)\|_2$, and $\epsilon$ is used as:
           $w^\pm = w \pm \epsilon\nabla_{w'}\mathcal L_{val}(w',\alpha)$. Value 0.01 of r is
           recommended for sufficiently accurate in the paper.

    References
    ----------
    _`[1]` H. Liu, K. Simonyan, Y. Yang, "DARTS: Differentiable Architecture Search",
     in ICLR, 2019.
    """

    def __init__(
            self,
            ul_objective: Callable[[Tensor, Tensor, Module, Module], Tensor],
            ul_model: Module,
            ll_objective: Callable[[Tensor, Tensor, Module, Module], Tensor],
            ll_model: Module,
            lower_learning_rate: float,
            update_ll_model_init: bool = False,
            r: float = 1e-2
    ):
        super(Onestage, self).__init__(ul_objective, ul_model, ll_model)
        self.ll_objective = ll_objective
        self.lower_learning_rate = lower_learning_rate
        self.update_initialization = update_ll_model_init
        self._r = r

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
               The validation data used for upper level problem optimization.

            validate_target: Tensor
               The labels of the samples in the validation data.

            auxiliary_model: _MonkeyPatchBase
                Wrapper of lower adapt_model encapsulated by module higher, has been optimized in lower
                optimization phase.

            train_data: Tensor
                The training data used for upper level problem optimization.

            train_target: Tensor
                The labels of the samples in the train data.

        Returns
        -------
        upper_loss: Tensor
           Returns the loss value of upper objective.
        """
        loss = self.ul_objective(validate_data, validate_target, self.ul_model, auxiliary_model)
        grad_x = torch.autograd.grad(loss, list(self.ul_model.parameters()), retain_graph=True)
        grad_y = torch.autograd.grad(loss, list(auxiliary_model.parameters()), retain_graph=self.update_initialization)

        dalpha = [v.data for v in grad_x]
        vector = [v.data for v in grad_y]
        implicit_grads = self._hessian_vector_product(vector, train_data, train_target)

        for g, ig in zip(dalpha, implicit_grads):
            g.sub_(self.lower_learning_rate, ig.data)

        if self.update_initialization:
            grads_lower = torch.autograd.grad(loss, list(auxiliary_model.parameters(time=0)))
            update_grads(grads_lower, self.ll_model)

        update_grads(dalpha, self.ul_model)

        return loss

    def _hessian_vector_product(
            self,
            vector,
            train_data,
            train_target
    ):
        """
        Built-in calculation function. Compute the first order approximation of
        the second-order derivative of upper variables.

        Parameters
        ----------
           train_data: Tensor
                The training data used for upper level problem optimization.

            train_target: Tensor
                The labels of the samples in the train data.

        Returns
        -------
        Tensor
           Returns the calculated first order approximation grads.
        """
        eta = self._r / torch.cat([x.view(-1) for x in vector]).norm()
        for p, v in zip(self.ll_model.parameters(), vector):
            p.data.add_(eta, v)  # w+
        loss = self.ll_objective(train_data, train_target, self.ul_model, self.ll_model)
        grads_p = torch.autograd.grad(loss, list(self.ul_model.parameters()))

        for p, v in zip(self.ll_model.parameters(), vector):
            p.data.sub_(2 * eta, v)  # w-
        loss = self.ll_objective(train_data, train_target, self.ul_model, self.ll_model)
        grads_n = torch.autograd.grad(loss, list(self.ul_model.parameters()))

        for p, v in zip(self.ll_model.parameters(), vector):
            p.data.add_(eta, v)  # w

        return [(x - y).div_(2 * eta) for x, y in zip(grads_p, grads_n)]
