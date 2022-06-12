from torch.nn import functional as F


def acc(out, target):
    pred = out.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item() / len(target)


def ll_o(data, target, ul_model, ll_model):
    out_f = ll_model(ul_model(data))
    loss_f = F.cross_entropy(out_f, target)
    return loss_f


def ul_o(data, target, ul_model, ll_model):
    out_F = ll_model(ul_model(data))
    loss_F = F.cross_entropy(out_F, target)
    return loss_F


def inner_o(data, target, model, params):
    return F.cross_entropy(model(data, params), target)


def outer_o(data, target, model, params):
    return F.cross_entropy(model(data, params), target)


def ul_o_iapttgm(data, target, ul_model, ll_model, time=-1):
    if time>0:
        out_F = ll_model(ul_model(data), params=ll_model.parameters(time=time))
        loss_F = F.cross_entropy(out_F, target)
    else:
        out_F = ll_model(ul_model(data))
        loss_F = F.cross_entropy(out_F, target)
    return loss_F