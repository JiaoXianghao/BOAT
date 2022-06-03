import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch import nn
import math


# def cross_entropy(pred, label, method="MetaInit"):
#     # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
#     assert method in ("MetaInit", "MetaRepr"), "Wrong value for argument method"
#     if method == "MetaInit":
#         return F.cross_entropy(logits=pred, labels=label)
# """    elif method == "MetaRepr":
#         return tf.reduce_mean(
#             tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
#         )
# """
#
#
# def mse(pred, label, method="MetaInit"):
#     assert method in ("MetaInit", "MetaRepr"), "Wrong value for argument method"
#     if method == "MetaInit":
#         return F.mse_loss(logits=pred, labels=label)
def accuary(out, target):
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc


def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def p_norm_reg(parameters, exp, epi):
    loss = 0
    for w in parameters:
        loss += (torch.norm(w, 2)+torch.norm(epi*torch.ones_like(w), 2))**(exp/2)
    return loss


def bias_reg_f(bias, params):
    # l2 biased regularization
    return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])


def distance_reg(output, label, params, hparams, reg_param):
    # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
    return F.cross_entropy(output, label) + reg_param * bias_reg_f(hparams, params)


def classification_acc(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item() / len(target)


def update_grads(grads, model):
    for p, x in zip(grads, model.parameters()):
        if x.grad is None:
            x.grad = p
        else:
            x.grad += p


def initialize(model):
    r"""
    Initializes the value of network variables.
    :param model:
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def draw_chart(val_losses, title):
    plt.plot(val_losses)
    plt.savefig(title, dpi=300)
