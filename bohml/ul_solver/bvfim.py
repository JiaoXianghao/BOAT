import torch
from .ul_optimizer import ULGrad
from bohml.utils.utils import loss_L2
from ..utils.utils import update_grads

from torch.nn import Module
from torch import Tensor
from typing import Callable


class BVFIM(ULGrad):
    r"""Calculation of the gradient of the upper adapt_model variables with BVFIM method

        Implements the UL optimization procedure of  Value-Function Best-
        Response (VFBR) type BLO methods, named i-level Value-Function-basedInterior-point
        Method(BVFIM) `[1]`_.

        A wrapper of lower adapt_model that has been optimized in the lower optimization will
        be used in this procedure.

        Note that this UL optimization module should only use bohml.ll_solver.BVFIM
        module to finish LL optimization procedure.

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

            ll_l2_reg: float, default=0.1
                Weight of L2 regularization term in the value function of the regularized
                LL problem, which is $\displaystyle f_\mu^* = \min_{y\in\mathbb{R}^n}
                f(x,y) + \frac{\mu_1}{2}\|y\|^2 + \mu_2$.

            ul_l2_reg: float, default=0.01
                Weight of L2 regularization term in the value function of the regularized
                UL problem, which is  which is $\displaystyle \varphi(x) = \min_{y\in\mathbb{R}^n} F(x,y)
                 + \frac{\theta}{2}\|y\|^2 - \tau\ln(f_\mu^*(x)-f(x,y))$.

            ul_ln_reg: float, default=10.
                Weight of the log-barrier penalty term in the value function of the regularized
                UL problem, as ul_l2_reg.

       References
       ----------
       _`[1]` R. Liu, X. Liu, X. Yuan, S. Zeng and J. Zhang, "A Value-Function-based
        Interior-point Method for Non-convex Bi-level Optimization", in ICML, 2021.
       """
    def __init__(
            self,
            ul_objective: Callable[[Tensor, Tensor, Module, Module], Tensor],
            ul_model: Module,
            ll_objective: Callable[[Tensor, Tensor, Module, Module], Tensor],
            ll_model: Module,
            ll_l2_reg: float = 0.1,
            ul_l2_reg: float = 0.01,
            ul_ln_reg: float = 10.
    ):
        super(BVFIM, self).__init__(ul_objective, ul_model, ll_model)
        self.ll_objective = ll_objective
        self.ll_l2_reg = ll_l2_reg
        self.ll_l2_reg = ul_l2_reg
        self.ul_ln_reg = ul_ln_reg

    def compute_gradients(
            self,
            validate_data: Tensor,
            validate_target: Tensor,
            auxiliary_model: Module,
            train_data: Tensor,
            train_target: Tensor,
            reg_decay: float
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

            reg_decay: float
                Weight decay coefficient of L2 regularization term and log-barrier
                penalty term. The value increases with the number of iterations.

        Returns
        -------
        upper_loss: Tensor
           Returns the loss value of upper objective.
        """
        loss_l2_z = self.ll_l2_reg / reg_decay * loss_L2(self.ll_model.parameters())
        loss_z_ = self.ll_objective(train_data, train_target, self.ul_model, self.ll_model)
        loss_z = loss_z_ + loss_l2_z

        loss_y_f_ = self.ll_objective(train_data, train_target, self.ul_model, auxiliary_model)
        loss_ln = self.ul_ln_reg / reg_decay * torch.log(loss_y_f_.item() + loss_z - loss_y_f_)

        loss_x_ = self.ul_objective(validate_data, validate_target, self.ul_model, auxiliary_model)
        loss_x = loss_x_ - loss_ln
        grads = torch.autograd.grad(loss_x, list(self.ul_model.parameters()))

        update_grads(grads, self.ul_model)

        return loss_x_
