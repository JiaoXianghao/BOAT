from .ll_optimizer import LLOptimize

from torch.nn import Module
from torch import Tensor
from typing import Callable
from higher.patch import _MonkeyPatchBase
from higher.optim import DifferentiableOptimizer


class Implicit(LLOptimize):
    r"""Lower level model optimization procedure

    Implement the LL model update process.

    The implemented lower level optimization procedure will optimize a wrapper of lower
     level model for further using in the following upper level optimization.

    Parameters
    ----------
        ll_objective: callable
            An optimization problem which is considered as the constraint of upper
            level problem.

            Callable with signature callable(state). Defined based on modeling of
            the specific problem that need to be solved. Computing the loss of LL
            problem. The state object contains the following:

            - "data"
                Data used in the upper optimization phase.
            - "target"
                Target used in the upper optimization phase.
            - "ul_model"
                Upper adapt_model of the bi-level adapt_model structure.
            - "ll_model"
                Lower adapt_model of the bi-level adapt_model structure.

        lower_loop: int
            Updating iterations over lower level optimization.

        ul_model: Module
            Upper adapt_model in a hierarchical adapt_model structure whose parameters will be
            updated with upper objective.

        ll_model: Module
            Lower adapt_model in a hierarchical adapt_model structure whose parameters will be
            updated with lower objective during lower-level optimization.
    """

    def __init__(
            self,
            ll_objective: Callable[[Tensor, Tensor, Module, Module], Tensor],
            lower_loop: int,
            ul_model: Module,
            ll_model: Module
    ):

        super(Implicit, self).__init__(ll_objective, lower_loop, ul_model, ll_model)

    def optimize(
        self,
        train_data: Tensor,
        train_target: Tensor,
        auxiliary_model: _MonkeyPatchBase,
        auxiliary_opt: DifferentiableOptimizer
    ):
        """
        Execute the lower optimization procedure with training data samples using lower
        objective. The passed in wrapper of lower adapt_model will be updated.

        Parameters
        ----------
            train_data: Tensor
                The training data used for LL problem optimization.

            train_target: Tensor
                The labels of the samples in the train data.

            auxiliary_model: _MonkeyPatchBase
                Wrapper of lower adapt_model encapsulated by module higher, will be optimized in lower
                optimization procedure.

            auxiliary_opt: DifferentiableOptimizer
                Wrapper of lower optimizer encapsulated by module higher, will be used in lower
                optimization procedure.

        Returns
        -------
        None
        """

        for lower_iter in range(self.lower_loop):
            lower_loss = self.ll_objective(train_data, train_target, self.ul_model, auxiliary_model)
            auxiliary_opt.step(lower_loss)
