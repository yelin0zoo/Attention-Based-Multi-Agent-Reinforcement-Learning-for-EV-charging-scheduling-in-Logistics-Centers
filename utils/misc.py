"""유틸: 타겟 업데이트, Gumbel-Softmax, 그래디언트 제어 등."""
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np


def soft_update(target, source, tau):
    """θ_target ← (1−τ)·θ_target + τ·θ_source"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def average_gradients(model):
    """분산 학습용 all-reduce 평균."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def init_processes(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def onehot_from_logits(logits, eps=0.0, dim=1):
    """argmax 원-핫 (eps>0이면 ε-greedy)."""
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Gumbel(0,1) = -log(-log(U))."""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, dim=1):
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=dim)


def gumbel_softmax(logits, temperature=1.0, hard=False, dim=1):
    """hard=True: Straight-Through (순전파 one-hot, 역전파 soft)."""
    y = gumbel_softmax_sample(logits, temperature, dim=dim)
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y = (y_hard - y).detach() + y
    return y


def firmmax_sample(logits, temperature, dim=1):
    if temperature == 0:
        return F.softmax(logits, dim=dim)
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)) / temperature
    return F.softmax(y, dim=dim)


def categorical_sample(probs, use_cuda=False):
    """반환: (int_acs[B,1], one_hot[B,n_actions])"""
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs


def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False


def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True


def sep_clip_grad_norm(parameters, max_norm, norm_type=2):
    """파라미터별 독립 그래디언트 클리핑 (통합 노름이 아닌 각각)."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    for p in parameters:
        if norm_type == float('inf'):
            p_norm = p.grad.data.abs().max()
        else:
            p_norm = p.grad.data.norm(norm_type)
        clip_coef = max_norm / (p_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)
