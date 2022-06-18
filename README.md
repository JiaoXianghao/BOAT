
# BOLV - A Bilevel Optimization Toolkit in Python for Learning and Vision Tasks
[![PyPI version](https://badge.fury.io/py/boml.svg)](https://badge.fury.io/py/boml)
[![Build Status](https://travis-ci.com/dut-media-lab/BOML.svg?branch=master)](https://travis-ci.com/dut-media-lab/BOML)
[![codecov](https://codecov.io/gh/JiaoXianghao/BOLV/branch/main/graph/badge.svg?token=8Y3OK2WPDE)](https://codecov.io/gh/JiaoXianghao/BOLV)
[![Documentation Status](https://readthedocs.org/projects/boml/badge/?version=latest)](https://boml.readthedocs.io/en/latest/?badge=latest)
![Language](https://img.shields.io/github/languages/top/dut-media-lab/boml?logoColor=green)
![Python version](https://img.shields.io/pypi/pyversions/boml)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

BOML is a modularized optimization library that unifies several ML algorithms into a common bilevel optimization framework. It provides interfaces to implement popular bilevel optimization algorithms, so that you could quickly build your own meta learning neural network and test its performance.

ReadMe.md contains brief introduction to implement meta-initialization-based and meta-feature-based methods in few-shot classification field. Except for algorithms which have been proposed, various combinations of lower level and upper level strategies are available. 

## Meta Learning 

Meta learning works fairly well when facing incoming new tasks by learning an initialization with favorable generalization capability. And it also has good performance even provided with a small amount of training data available, which gives birth to various solutions for different application such as few-shot learning problem.

We present a general bilevel optimization paradigm to unify different types of meta learning approaches, and the mathematical form could be summarized as below:<br>

<div align=center>
  
![Bilevel Optimization Model](https://github.com/dut-media-lab/BOML/blob/master/figures/p1.png)
</div>

## Generic Optimization Routine
Here we illustrate the generic optimization process and hierarchically built strategies in the figure, which could be quikcly implemented in the following example.<br>

<div align=center>
  
![Optimization Routine](https://github.com/dut-media-lab/BOML/blob/master/figures/p2.png)
</div>

## Documentation 
For more detailed information of basic function and construction process, please refer to our [Documentation](https://boml.readthedocs.io) or[Project Page](https://dut-media-lab.github.io/BOML/). Scripts in the directory named test_script are useful for constructing general training process.

Here we give recommended settings for specific hyper paremeters to quickly test performance of popular algorithms.

## Running examples
### Start from loading data
```python
import boml
from boml import utils
from test_script.script_helper import *

dataset = boml.load_data.meta_omniglot(
    std_num_classes=args.classes,
    examples_train=args.examples_train,
    examples_test=args.examples_test,
)
# create instance of BOMLExperiment for ong single task
ex = boml.BOMLExperiment(dataset)

```

### Build network structure and define parameters for meta-learner and base-learner

```python
boml_ho = boml.BOLVOptimizer(
    method="MetaInit", inner_method="Simple", outer_method="Simple"
)
meta_learner = boml_ho.meta_learner(_input=ex.x, dataset=dataset, meta_model="V1")
ex.adapt_model = boml_ho.base_learner(_input=ex.x, meta_learner=meta_learner)
``` 
### Define LL objectives and LL calculation process

```python
loss_inner = utils.cross_entropy(pred=ex.adapt_model.out, label=ex.y)
accuracy = utils.classification_acc(pred=ex.adapt_model.out, label=ex.y)
inner_grad = boml_ho.ll_problem(
    inner_objective=loss_inner,
    learning_rate=args.lr,
    T=args.T,
    experiment=ex,
    var_list=ex.adapt_model.var_list,
)
```
### Define UL objectives and UL calculation process

```python
loss_outer = utils.cross_entropy(pred=ex.adapt_model.re_forward(ex.x_).out, label=ex.y_)  # loss function
boml_ho.ul_problem(
    outer_objective=loss_outer,
    meta_learning_rate=args.meta_lr,
    inner_grad=inner_grad,
    meta_param=tf.get_collection(boml.extension.GraphKeys.METAPARAMETERS),
)
```
### Aggregate all the defined operations
```python
# Only need to be called once after all the tasks are ready
boml_ho.aggregate_all()
```
### Meta training iteration
```python
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    for itr in range(args.meta_train_iterations):
        # Generate the feed_dict for calling run() everytime
        train_batch = BatchQueueMock(
            dataset.train, 1, args.meta_batch_size, utils.get_rand_state(1)
        )
        tr_fd, v_fd = utils.feed_dict(train_batch.get_single_batch(), ex)
        # Meta training step
        boml_ho.run(tr_fd, v_fd)
        if itr % 100 == 0:
            print(sess.run(loss_inner, utils.merge_dicts(tr_fd, v_fd)))
```

## Related Methods 
 - [Hyperparameter optimization with approximate gradient(HOAG)](https://arxiv.org/abs/1602.02355)
 - [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks(MAML)](https://arxiv.org/abs/1703.03400)
 - [On First-Order Meta-Learning Algorithms(FMAML)](https://arxiv.org/abs/1703.03400)
 - [Meta-SGD: Learning to Learn Quickly for Few-Shot Learning(Meta-SGD)](https://arxiv.org/pdf/1707.09835.pdf)
 - [Bilevel Programming for Hyperparameter Optimization and Meta-Learning(RHG)](http://export.arxiv.org/pdf/1806.04910)
 - [Truncated Back-propagation for Bilevel Optimization(TG)](https://arxiv.org/pdf/1810.10667.pdf)
 - [Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace(MT-net)](http://proceedings.mlr.press/v80/lee18a/lee18a.pdf)
 - [Meta-Learning with warped gradient Descent(WarpGrad))](https://arxiv.org/abs/1909.00025)
 - [DARTS: Differentiable Architecture Search(DARTS)](https://arxiv.org/pdf/1806.09055.pdf)
 - [A Generic First-Order Algorithmic Framework for Bi-Level Programming Beyond Lower-Level Singleton(BDA)](https://arxiv.org/pdf/2006.04045.pdf)



## License

MIT License

Copyright (c) 2020 Yaohua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



