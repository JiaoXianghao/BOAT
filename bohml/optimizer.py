import time
import higher
import copy

from bohml.init_optimize.init import Init

from typing import List, Callable, Union, Dict
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from bohml.utils.model.meta_model import MetaModel
from bohml.utils.utils import initialize

importlib = __import__("importlib")
ll_grads = importlib.import_module("bohml.ll_solver")
ul_grads = importlib.import_module("bohml.ul_solver")


class BOHMLOptimizer(object):
    r""" Wrapper for performing bi-level optimization and gradient-based initialization optimization

    BOHMLOptimizer is the wrapper of Bi-Level Optimization(BLO) and Initialization Optimization(Initialization-based
    EGBR) process which builds LL, UL and Initialization problem solver with corresponding method modules
    and uses in training phase. The optimization process could also be done by using methods packages directly.

    Parameters
    ----------
    method: str
        Define basic method for following training process, it should be included in
        ['Initial', 'Feature']. 'Initial' type refers to meta-learning optimization
        strategy, including methods like 'MAML, FOMAML, TNet, WarpGrad, L2F'; 'Feature'
        type refers to bi-level optimization strategy, includes methods like 'BDA, RHG,
        Truncated RHG, Onestage, BVFIM, IAPTT-GM, LS, NS, GN, BVFIM'.

    ll_method: str, default=None
        method chosen for solving LL problem, including ['Dynamic' | 'Implicit' | 'BVFIM'].

    ul_method: str, default=None
        Method chosen for solving UL problem, including ['Recurrence','Onestage' | 'LS','NS',
        'GN' | 'BVFIM'].

    ll_objective: callable, default=None
        An optimization problem which is considered as the constraint of UL problem.

        Callable with signature callable(state). Defined based on modeling of
        the specific problem that need to be solved. Computing the loss of upper
        problem. The state object contains the following:

        - "data"
            Data used in the LL optimization phase.
        - "target"
            Target used in the LL optimization phase.
        - "ul_model"
            UL adapt_model of the bi-level adapt_model structure.
        - "ll_model"
            LL adapt_model of the bi-level adapt_model structure.

    ul_objective: callable, default=None
        The main optimization problem in a hierarchical optimization problem.

        Callable with signature callable(state). Defined based on modeling of
        the specific problem that need to be solved. Computing the loss of upper
        problem. The state object contains the following:

        - "data"
            Data used in the UL optimization phase.
        - "target"
            Target used in the UL optimization phase.
        - "ul_model"
            Ul adapt_model of the bi-level adapt_model structure.
        - "ll_model"
            LL adapt_model of the bi-level adapt_model structure.
        - "time" (optional)
            Parameter for IAPTT-GM method, denote that ll model forward with
            the variables of 'time' times loop.

    inner_objective: callable, default=None
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

    outer_objective: callable, default=None
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

    ll_model: Module, default=None
        The adapt_model whose parameters will be updated during upper-level optimization.

    ul_model: Module, default=None
        Upper adapt_model in a hierarchical adapt_model structure whose parameters will be
        updated with upper objective.

    meta_model: Module, default=None  # todo
        Model whose initial value will be optimized. If choose MAML method to optimize, any user-defined
        torch nn.Module could be used as long as the definition of forward() function meets the standard;
        but if choose other derived methods, internally defined bohml.utils.adapt_model.meta_model should be used
        for related additional modules.

    total_iters: int, default=60000
        Total iterations of the experiment, used to set weight decay.
    """

    def __init__(
            self,
            method: str,
            ll_method: str = None,
            ul_method: str = None,
            ll_objective: Callable[[Tensor, Tensor, Module, Module], Tensor] = None,
            ul_objective: Union[Callable[[Tensor, Tensor, Module, Module], Tensor],
                                Callable[[Tensor, Tensor, Module, Module, int], Tensor]] = None,
            inner_objective: Callable[[Tensor, Tensor, Module, Dict[str, List[Parameter]]], Tensor] = None,
            outer_objective: Callable[[Tensor, Tensor, Module, Dict[str, List[Parameter]]], Tensor] = None,
            ll_model: Module = None,
            ul_model: Module = None,
            meta_model: MetaModel = None,
            total_iters: int = 60000
    ):
        super(BOHMLOptimizer, self).__init__()
        assert method in ("Feature", "Initial"), (
            "initialize method argument, should be in list [MetaRepr,MetaInitl] "
            "Feature based methods include [BDA,RAD,TRAD,IAPTT-GM,Onestage-RAD,BVFIM,LS,NS,GN],"
            "Initial based methods include [MAML,MSGD,MT-net,Warp-grad,L2F]"
        )
        self._method = method

        if self._method == "Feature":
            assert ll_method is not None, "'ll_method' shouldn't be None,select from [Dynamic | Implicit | BVFIM]"
            assert ll_method in ("Dynamic", "Implicit", "BVFIM"),\
                "invalid method argument, select from list [Dynamic | Implicit | BVFIM]"
            self._ll_method = ll_method

            assert ul_method is not None, (
                "'ul_method' shouldn't be None,select from [Recurrence,Onestage | LS,NS,GN | BVFIM]"
            )
            if self._ll_method == "Dynamic":
                assert ul_method in ("Recurrence", "Onestage"), (
                    "Invalid method argument, select from list [Recurrence,Onestage]"
                )
            elif self._ll_method == "Implicit":
                assert ul_method in ("LS", "NS", "GN"), (
                    "Invalid method argument, select from list [LS,NS,GN]"
                )
            else:
                assert ul_method == "BVFIM", (
                    "Invalid combination of inner and upper strategies, "
                    "The bilevel BVFIM strategy should choose 'BVFIM' as UL optimization strategy"
                )
            self._ul_method = ul_method

            assert ll_objective is not None, (
                "'ll_objective shouldn't be None, "
                "define according to the form objective(data, target, ul_model, ll_model)"
            )
            assert ul_objective is not None, (
                "'ul_objective shouldn't be None, "
                "define according to the form objective(data, target, ul_model, ll_model)"
            )
            self._ll_objective = ll_objective
            self._ul_objective = ul_objective

            assert ll_model is not None, "'ll_model' shouldn't be None"
            assert ul_model is not None, "'ul_model' shouldn't be None"
            self._ll_model = ll_model
            self._ul_model = ul_model
            initialize(self._ll_model)
            initialize(self._ul_model)

        else:
            assert inner_objective is not None, (
                "'inner_objective shouldn't be None, "
                "define according to the form objective(data, target, meta_model, updated_weights)"
            )
            assert outer_objective is not None, (
                "'outer_objective shouldn't be None, "
                "define according to the form objective(data, target, meta_model, updated_weights)"
            )
            self._inner_objective = inner_objective
            self._outer_objective = outer_objective

            assert meta_model is not None, "'meta_model' shouldn't be None"
            if isinstance(meta_model, MetaModel):
                self._meta_model = meta_model
                initialize(self._meta_model)
            else:
                raise TypeError(
                    'Invalid type of "meta_model" argument, initialization-based EGBR method should'
                    ' be implemented with model wrapper defined by bohml.utils.model.meta_model.MetaModel'
                )

        self._ll_problem_solver = None
        self._ul_problem_solver = None
        self._meta_problem_solver = None

        self._auxiliary_model = None
        self._auxiliary_opt = None

        self._lower_opt = None
        self._lower_init_opt = None
        self._lower_learning_rate = 0.0
        self._upper_opt = None
        self._upper_learning_rate = 0.0
        self._meta_opt = None
        self._lr_scheduler = None  # todo

        self._meta_iters = total_iters
        self._param_dict = dict()

    def build_ll_solver(
            self,
            lower_loop: int,
            ll_optimizer: Optimizer,
            truncate_iter: int = 0,
            truncate_max_loss_iter: bool = False,
            alpha_init=0.0,
            alpha_decay: float = 0.0,
            y_loop: int = 5,
            ll_l2_reg: float = 0.1,
            ul_l2_reg: float = 0.01,
            ul_ln_reg: float = 10.,
            reg_decay: bool = True,
            **kwargs
    ):
        """
        Build LL-problem solver with bohml.lower_optimizer module,
        which will optimize lower adapt_model for further using in UL optimization
        procedure. Setting the value of parameters according to the selected method.

        Details of parameter settings for each particular method are available in the specific
        method module of bohml.ul_solver

        Parameters
        ----------
            lower_loop: int
                The total number of iterations for lower gradient descent optimization.

            ll_optimizer: Optimizer
                Optimizer of lower adapt_model, defined outside this module and will be used
                in LL optimization procedure.

            update_ll_model_with_step_num: int, default=0
                Whether to update lower adapt_model variables after LL optimization. Default
                value 0 means that lower adapt_model will maintain initial state after LL optimization
                process. If set this parameter to a positive integer k, then the lower
                adapt_model will save the updated results of step k of the LL optimization loop.
                Setting it when experiment doesn't have fine-tune stage.

            truncate_iter: int, default=0
                Specific parameter for Truncated Reverse AD method, defining number of
                iterations to truncate in the back propagation process during lower
                optimizing.

            truncate_max_loss_iter: bool, default=False
                Specific parameter for IAPTT-GM method,if set True then will use PTT method to truncate
                reverse trajectory during ll optimization.

            alpha_init: float, default=0.0
                Specify parameter for BDA method. The aggregation parameter for Bi-level descent
                aggregation method, where alpha ∈ (0, 1) denotes the ratio of lower objective
                to upper objective during lower optimizing.

            alpha_decay: float, default=0.0
                Specify parameter for BDA method. Weight decay coefficient of aggregation parameter alpha.
                The decay rate will accumulate with ll optimization procedure.

            z_loop: int, default=5
                Specify parameter for BVFIM method. Num of steps to obtain a low LL problem value, i.e.
                 optimize LL variable with LL problem. Regarded as $T_z$ in the paper.

            y_loop: int, default=5
                Specify parameter for BVFIM method. Num of steps to obtain a optimal LL variable under the
                LL problem value obtained after lower_loop, i.e. optimize the updated LL variable with UL
                 problem. Regarded as Regarded as $T_y$ in the paper.

            ll_l2_reg: float, default=0.1
                Specify parameter for BVFIM method. Weight of L2 regularization term in the value
                function of the regularizedLL problem. Referring to module bohml.ul_solver.bvfim
                for more details.

            ul_l2_reg: float, default=0.01
                Specify parameter for BVFIM method. Weight of L2 regularization term in the
                value function of the regularized UL problem. Referring to module
                bohml.ul_solver.bvfim for more details.

            ul_ln_reg: float, default=10.
                Specify parameter for BVFIM method. Weight of the log-barrier penalty term in the
                value function of the regularized UL problem. Referring to module bohml.ul_solver.bvfim
                for more details.

            reg_decay: bool, default=True
                Specify parameter for BVFIM method. Whether to use weight decay coefficient of
                 L2 regularization term and log-barrier penalty term.
        """

        if self._ll_method == 'Dynamic':
            if self._ul_method == 'Onestage':
                assert lower_loop == 1, "One-stage method requires one gradient step to optimize task parameters."
                assert truncate_iter == 0 and not truncate_max_loss_iter, \
                    "One-stage method doesn't need trajectory truncation."
            assert truncate_iter == 0 or not truncate_max_loss_iter, \
                "Only one of the IAPTT-GM and TRAD methods could be chosen."
            assert (
                0.0 <= alpha_init <= 1.0
            ), "Parameter 'alpha' used in method BDA should be in the interval (0,1)."
            if alpha_init > 0.0:
                assert (0.0 < alpha_decay <= 1.0), \
                    "Parameter 'alpha_decay' used in method BDA should be in the interval (0,1)."
            assert (truncate_iter < lower_loop), "The value of 'truncate_iter' shouldn't be greater than 'lower_loop'."

        self._lower_opt = ll_optimizer
        self._lower_learning_rate = self._lower_opt.defaults['lr']

        # build LL problem solver
        if self._ll_method == "Dynamic":
            kwargs['ul_objective'] = self._ul_objective
            self._param_dict['truncate_max_loss_iter'] = truncate_max_loss_iter
            if truncate_iter > 0:
                kwargs['truncate_iters'] = truncate_iter
                kwargs['ll_opt'] = self._lower_opt
            kwargs['truncate_max_loss_iter'] = truncate_max_loss_iter
            kwargs['alpha'] = alpha_init
            if alpha_init > 0.0:
                kwargs['alpha_decay'] = alpha_decay
        elif self._ll_method == "BVFIM":
            kwargs['ul_objective'] = self._ul_objective
            kwargs['y_loop'] = y_loop
            kwargs['ll_opt'] = self._lower_opt
            kwargs['ll_l2_reg'] = ll_l2_reg
            kwargs['ul_l2_reg'] = ul_l2_reg
            kwargs['ul_ln_reg'] = ul_ln_reg
            self._param_dict['reg_decay'] = reg_decay

        self._ll_problem_solver = getattr(
            ll_grads, "%s" % self._ll_method
        )(ll_objective=self._ll_objective,
          ll_model=self._ll_model,
          ul_model=self._ul_model,
          lower_loop=lower_loop,
          **kwargs)

        return self

    def build_ul_solver(
            self,
            ul_optimizer: Optimizer,
            update_ll_model_init: bool = False,
            k: int = 10,
            tolerance: float = 1e-10,  # =  lambda _k: 0.1 * (0.9 ** _k),
            r: float = 1e-2,
            ll_l2_reg: float = 0.1,
            ul_l2_reg: float = 0.01,
            ul_ln_reg: float = 10.,
            **kwargs
    ):
        """
        Setting up UL optimization module. Select desired method through given parameters
        and set related experiment parameters.

        Details of parameter settings for each particular method are available in the specific
        method module of bohml.ul_solver.

        Parameters
        ----------
            ul_optimizer: Optimizer
                Optimizer of upper adapt_model, defined outside this module and will be used
                in UL optimization procedure.

            update_ll_model_init: bool, default=False
               Specific parameter for Dynamic method. If set True, the initial value of ll model will be updated after this iteration.

            k: int, default=10
                Specific parameter for Implicit method. The maximum number of conjugate gradient iterations.

            tolerance: float, default=1e-10
                Specific parameter for Implicit method. End the method earlier when the norm of the
                residual is less than tolerance.

            r: float, default=1e-2
                Parameter for One-stage Recurrence method and used to adjust scalar epsilon. Value 0.01 of r is
                recommended for sufficiently accurate in the paper. Referring to module
                bohml.ul_solver.onestage for more details.

            ll_l2_reg: float, default=0.1
                Specify parameter for BVFIM method. Weight of L2 regularization term in the value
                function of the regularizedLL problem. Referring to module bohml.ul_solver.bvfim
                for more details.

            ul_l2_reg: float, default=0.01
                Specify parameter for BVFIM method. Weight of L2 regularization term in the
                value function of the regularized UL problem. Referring to module
                bohml.ul_solver.bvfim for more details.

            ul_ln_reg: float, default=10.
                Specify parameter for BVFIM method. Weight of the log-barrier penalty term in the
                value function of the regularized UL problem. Referring to module bohml.ul_solver.bvfim
                for more details.
        """
        if update_ll_model_init:
            assert self._ll_method == "Dynamic", \
                "Choose 'Dynamic' as ll method if you want to use initialization auxiliary."

        self._upper_opt = ul_optimizer
        self._upper_learning_rate = self._upper_opt.defaults['lr']

        if self._ll_method == "Dynamic":
            if update_ll_model_init:
                self._lower_init_opt = copy.deepcopy(self._lower_opt)
                self._lower_init_opt.param_groups[0]['lr'] = self._upper_opt.param_groups[0]['lr']
            kwargs['update_ll_model_init'] = update_ll_model_init
            self._param_dict['update_ll_model_init'] = update_ll_model_init
            if self._ul_method == "Onestage":
                kwargs['ll_objective'] = self._ll_objective
                kwargs['lower_learning_rate'] = self._lower_learning_rate
                kwargs['r'] = r
            else:
                kwargs['truncate_max_loss_iter'] = self._param_dict['truncate_max_loss_iter']
        elif self._ll_method == "Implicit":
            kwargs['ll_objective'] = self._ll_objective
            if self._ul_method == "LS" or self._ul_method == "NS":
                kwargs['lower_learning_rate'] = self._lower_learning_rate
                kwargs['k'] = k
                kwargs['tolerance'] = tolerance
        elif self._ul_method == "BVFIM":
            kwargs['ll_objective'] = self._ll_objective
            kwargs['ll_l2_reg'] = ll_l2_reg
            kwargs['ul_l2_reg'] = ul_l2_reg
            kwargs['ul_ln_reg'] = ul_ln_reg

        self._ul_problem_solver = getattr(
            ul_grads, "%s" % self._ul_method
        )(ul_objective=self._ul_objective,
          ll_model=self._ll_model,
          ul_model=self._ul_model,
          **kwargs)

        return self

    def build_meta_solver(
            self,
            meta_optimizer: Optimizer,
            inner_loop: int = 5,
            inner_learning_rate: float = 0.1,
            use_second_order: bool = True,
            learn_lr: bool = False,
            use_t: bool = False,
            use_warp: bool = False,
            use_forget: bool = False
    ):
        """
        Setting up meta-learning optimization module. Select desired method through given parameters
        and set set related experiment parameters.

        Note that among three methods MT-net, Warpgrad and L2F, only one can be used; while First-order
        and MSGD can be combined with others.

        Parameters
        ----------
            meta_optimizer: Optimizer
                The optimizer used to update initial values of meta adapt_model after
                an iteration.

            inner_loop: int, default=5
                Num of inner optimization steps.

            inner_learning_rate: float, default=0.01
                Step size for inner optimization.

            use_second_order: bool, default=True
                Optional argument, whether to calculate precise second-order gradients during inner-loop.

            learn_lr: bool, default=False
                Optional argument, whether to update inner learning rate during outer optimization,
                i.e. use MSGD method.

            use_t: bool, default=False
                Optional argument, whether to using T-layers during optimization,i.e. use MT-net method.

            use_warp: bool, default=False
                Optional argument, whether to using warp modules during optimization,i.e. use Warp-grad method.

            use_forget: bool, default=False
                Optional argument, whether to add attenuation to each layers, i.e. use L2F method.

            enable_inner_loop_optimizable_bn_params: bool, default=False
                Parameter for L2F method. When use L2F, whether to add the attenuation operation to
                the batch-norm modules.

        """

        if use_t:
            assert not use_warp and not use_forget, \
                "MT-net method has been chosen, can't use other methods."
        elif use_warp:
            assert not use_t and not use_forget, \
                "Warp-grad method has been chosen, can't use other methods."
        elif use_forget:
            assert not use_warp and not use_t, \
                "L2F method has been chosen, can't use other methods."

        self._meta_opt = meta_optimizer
        self._meta_problem_solver = Init(self._meta_model, self._inner_objective,
                                         self._outer_objective, inner_learning_rate,
                                         inner_loop, use_second_order,
                                         learn_lr, use_t, use_warp, use_forget)

    def run_iter(
            self,
            train_data_batch: Tensor,
            train_target_batch: Tensor,
            validate_data_batch: Tensor,
            validate_target_batch: Tensor,
            current_iter: int,
            forward_with_whole_batch: bool = True
    ):
        """
        Run an iteration with a data batch and updates the parameters of upper adapt_model or meta-adapt_model.

        Parameters
        ----------
            train_data_batch: Tensor
                A batch of train data,which is used during lower optimizing.

            train_target_batch: Tensor
                A batch of train target,which is used during lower optimizing.

            validate_data_batch: Tensor
                A batch of test data,which is used during upper optimizing.

            validate_target_batch: Tensor
                A batch of test target,which is used during upper optimizing.

            current_iter: int
                The num of current iter.

            forward_with_whole_batch: bool, default=True
                Whether to feed in the whole data batch when doing forward propagation.
                When setting to False, each single data in the batch will be fed into adapt_model
                during this iteration. This useful for some experiment having special setting,
                like few-shot learning.

        Returns
        -------
        validation_loss: Tensor
            Returns the value of validation loss value.
        """
        kwargs_lower = {}
        kwargs_upper = {}
        losses = 0.0
        forward_time = 0.0
        backward_time = 0.0
        val_acc = 0.0

        if forward_with_whole_batch:
            train_data_batch = [train_data_batch]
            train_target_batch = [train_target_batch]
            validate_data_batch = [validate_data_batch]
            validate_target_batch = [validate_target_batch]

        for t_idx, (train_x, train_y, val_x, val_y) in enumerate(zip(train_data_batch, train_target_batch,
                                                                 validate_data_batch, validate_target_batch)):
            if self._method == "Initial":
                loss = self._meta_problem_solver.optimize(train_x, train_y, val_x, val_y)
                losses += loss.item()

            else:
                if self._ll_method == "BVFIM":
                    forward_time = time.time()
                    reg_decay = float(self._param_dict['reg_decay']) * current_iter + 1
                    auxiliary_model = copy.deepcopy(self._ll_model)
                    auxiliary_opt = torch.optim.SGD(auxiliary_model.parameters(), lr=0.01)
                    out, auxiliary_model = self._ll_problem_solver.optimize(train_x, train_y, auxiliary_model, auxiliary_opt,
                                                     val_x, val_y, reg_decay)
                    # val_acc += accuary(out, val_y) / 4
                    forward_time = time.time() - forward_time
                    backward_time = time.time()
                    loss = self._ul_problem_solver.compute_gradients(val_x, val_y, auxiliary_model, train_x,
                                                                     train_y, reg_decay)
                    backward_time = time.time() - backward_time
                    initialize(self._ll_model)
                else:
                    with higher.innerloop_ctx(self._ll_model, self._lower_opt,
                                              copy_initial_weights=False) as (auxiliary_model, auxiliary_opt):
                        # LL level problem optimizing
                        forward_time = time.time()
                        if self._ll_method == "Dynamic":
                            kwargs_lower['validate_data'] = val_x
                            kwargs_lower['validate_target'] = val_y
                        pmax = self._ll_problem_solver.optimize(train_x, train_y, auxiliary_model, auxiliary_opt,
                                                                **kwargs_lower)
                        forward_time = time.time() - forward_time

                        # UL problem optimizing
                        backward_time = time.time()
                        if self._ll_method == "Dynamic":
                            if self._param_dict['truncate_max_loss_iter']:
                                kwargs_upper['max_loss_iter'] = pmax
                            elif self._ul_method == "Onestage":
                                kwargs_upper['train_data'] = train_x
                                kwargs_upper['train_target'] = train_y
                        elif self._ll_method == 'Implicit':
                            kwargs_upper['train_data'] = train_x
                            kwargs_upper['train_target'] = train_y
                        loss = self._ul_problem_solver.compute_gradients(val_x, val_y, auxiliary_model, **kwargs_upper)
                        backward_time = time.time() - backward_time
            losses += loss.item()

        batch_size = 1
        # update adapt_model parameters
        if self._method == "Initial":
            if batch_size > 1:
                for x in self._meta_model.parameters():
                    x.grad = x.grad / batch_size
            self._meta_opt.step()
            self._meta_opt.zero_grad()
        else:
            if self._ll_method == "Dynamic":
                if self._param_dict['update_ll_model_init']:
                    for x in self._ll_model.parameters():
                        x.grad = x.grad / batch_size
                    self._lower_init_opt.step()
                    self._lower_init_opt.zero_grad()
            if not forward_with_whole_batch and batch_size > 1:
                for x in self._ul_model.parameters():
                    x.grad = x.grad / batch_size

            self._upper_opt.step()
            self._upper_opt.zero_grad()

        return losses / batch_size, forward_time, backward_time

    # @property
    # def meta_model(self):
    #     """
    #     :return: the created BMLNet object
    #     """
    #     return self._meta_model
    #
    # @property
    # def outergradient(self):
    #     """
    #     :return: the outergradient object underlying this wrapper.
    #     """
    #     return self._outer_gradient
    #
    # @property
    # def innergradient(self):
    #     """
    #     :return: the innergradient object underlying this wrapper.
    #     """
    #     return self._inner_gradient
    #
    # @property
    # def learning_rate(self):
    #     """
    #     :return: the outergradient object underlying this wrapper.
    #     """
    #     return self._learning_rate
    #
    # @property
    # def meta_learning_rate(self):
    #     """
    #     :return: the outergradient object underlying this wrapper.
    #     """
    #     return self._meta_learning_rate
    #
    # @property
    # def method(self):
    #     """
    #     :return: the method for whole algorithm.
    #     """
    #     return self._method
    #
    # @property
    # def param_dict(self):
    #     """
    #     :return: dict that holds hyper_params used in the inner optimization process.
    #     """
    #     return self._param_dict
    #
    # @property
    # def inner_objectives(self):
    #     return self._outer_gradient.inner_objectives
