import abc


class LLOptimize(object):
    def __init__(
            self,
            ll_objective,
            lower_loop,
            ul_model,
            ll_model
    ) -> None:
        r"""Initialize the optimizer with the state of an existing optimizer.

        Args:
            other: an existing optimizer instance.
            reference_params: an iterable over the parameters of the original
                adapt_model.
            fmodel (optional): a patched stateless module with a view on
                weights.
            device (optional): the device to cast state tensors to.
        """
        r"""Perform a adapt_model update.

        This would be used by replacing the normal sequence::

            opt.zero_grad()
            loss.backward()
            opt.step()
        with::
            diffopt.step(loss)
        Args:
            loss: the loss tensor.
            params (optional): the parameters with regard to which we measure
                the loss. These must be provided if the differentiable optimize
                are provided, they will overwrite the params of the encapsulated
                adapt_model.
            override (optional): a dictionary mapping optimizer settings (i.e.


        Returns:
            The updated parameters, which will individually have ``grad_fn``\ s
            of their own. If the optimizer has an encapsulated patched adapt_model,
            its view over its own fast weights will be updated with these
            params.
        """
        self.ll_objective = ll_objective
        self.lower_loop = lower_loop
        self.ul_model = ul_model
        self.ll_model = ll_model

    @abc.abstractmethod
    def optimize(self, **kwargs):
        pass


# _OptMappingType = _typing.Dict[_torch.optim.BOLVOptimizer, _typing.
#                                Type[DifferentiableOptimizer]]
# _opt_mapping: _OptMappingType = {
#     _torch.optim.Adadelta: DifferentiableAdadelta,
#     _torch.optim.Adagrad: DifferentiableAdagrad,
#     _torch.optim.Adam: DifferentiableAdam,
#     _torch.optim.Adamax: DifferentiableAdamax,
#     _torch.optim.ASGD: DifferentiableASGD,
#     _torch.optim.RMSprop: DifferentiableRMSprop,
#     _torch.optim.Rprop: DifferentiableRprop,
#     _torch.optim.SGD: DifferentiableSGD,
# }
#
# def register_optim(
#     optim_type: _torch.optim.BOLVOptimizer,
#     diff_optim_type: _typing.Type[DifferentiableOptimizer]
# ) -> None:
#     r"""Registers a new optimizer type for use with higher functions.
#
#     Args:
#         optim_type: the type of a new optimizer, assumed to be an instance of
#             ``torch.optim.BOLVOptimizer``.
#         diff_optim_type: the type of a new differentiable optimizer, assumed to
#             be an instance of ``higher.optim.DifferentiableOptimizer`` with
#             functionally equivalent logic to ``optim_type``.
#     """
#     _opt_mapping[optim_type] = diff_optim_type
