"""
Referred from https://github.com/lx10077/rlpy/blob/master/utils/torchs.py
"""

import torch
import math
import numpy as np


# --- Pytorch operation shortcuts

use_gpu = torch.cuda.is_available()


def from_numpy(np_array, gpu=False):
    return torch.from_numpy(np_array).cuda() if gpu else torch.from_numpy(np_array)


def normal(shape, gpu=False, seed=0):
    torch.manual_seed(seed)
    matrix = torch.zeros(*shape)
    if gpu:
        torch.cuda.manual_seed(seed)
        matrix = matrix.cuda()
    matrix.normal_(std=1/math.sqrt(shape[0]))
    return matrix


def identity(dimension, gpu=False):
    matrix = torch.eye(dimension)
    if gpu:
        matrix = matrix.cuda()
    return matrix


def normal_like(matrix_tenser, gpu=False, seed=0):
    torch.manual_seed(seed)
    matrix = torch.zeros_like(matrix_tenser)
    if gpu:
        torch.cuda.manual_seed(seed)
        matrix = matrix.cuda()
    m = matrix_tenser.shape[0]
    matrix.normal_(std=1/math.sqrt(m))
    return matrix


def ones(shape, gpu=False, **kwargs):
    return torch.ones(*shape, **kwargs).cuda() if use_gpu and gpu else torch.ones(*shape)


def zeros(shape, gpu=False, **kwargs):
    return torch.zeros(*shape, **kwargs).cuda() if use_gpu and gpu else torch.zeros(*shape)


def one_hot(x, n):
    assert x.dim() == 2, "Incompatible dim {:d} for input. Dim must be 2.".format(x.dim())
    one_hot_x = torch.zeros(x.size(0), n)
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x


def np_to_tensor(nparray):
    assert isinstance(nparray, np.ndarray)
    return torch.from_numpy(nparray)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = t1.unsqueeze(2).unsqueeze(3).repeat(1, t2_height, t2_width, 1).view(out_height, out_width)
    return expanded_t1 * tiled_t2


# --- Getting and setting operations

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_state_dict(file):
    try:
        pretrain_state_dict = torch.load(file)
    except AssertionError:
        pretrain_state_dict = torch.load(file, map_location=lambda storage, location: storage)
    return pretrain_state_dict


def get_out_dim(module, indim):
    if isinstance(module, list):
        module = torch.nn.Sequential(*module)
    fake_input = torch.zeros(indim).unsqueeze(0)
    output_size = module(fake_input).view(-1).size()[0]
    return output_size


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def assign_params(from_model, to_model):
    if from_model is not None and to_model is not None:
        params = get_flat_params_from(from_model)
        set_flat_params_to(to_model, params)
    return


def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(zeros(param.data.view(-1).shape))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


# --- Computing and estimating operations

def get_grad_dict(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = dict()
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads[i] = zeros(param.data.view(-1).shape)
        else:
            out_grads[i] = grads[j]
            j += 1

    for param in params:
        param.grad = None
    return out_grads


def get_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(zeros(param.data.view(-1).shape))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads


def get_tensor_info(tensor, precision=2):
    return {'mean': round(tensor.mean().item(), precision),
            'max': round(tensor.max().item(), precision),
            'min': round(tensor.min().item(), precision),
            'std': round(tensor.std().item(), precision)
            }
