"""
유틸리티 함수 모듈
MAAC 학습에 필요한 다양한 헬퍼 함수들을 제공
- 네트워크 파라미터 업데이트 (소프트/하드)
- 분산 학습 관련 함수
- Gumbel-Softmax 샘플링 (이산 행동의 미분 가능한 샘플링)
- 그래디언트 제어 함수
"""
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist  # 분산 학습 모듈
from torch.autograd import Variable
import numpy as np


def soft_update(target, source, tau):
    """
    DDPG 스타일 소프트 업데이트 (지수 이동 평균)
    타겟 네트워크의 파라미터를 소스 네트워크 방향으로 천천히 이동시킴

    공식: θ_target = (1 - τ) * θ_target + τ * θ_source

    τ가 작을수록 (예: 0.001) 타겟이 천천히 변하여 학습이 안정적
    τ = 1이면 하드 업데이트와 동일 (완전 복사)

    인자:
        target (torch.nn.Module): 업데이트할 타겟 네트워크
        source (torch.nn.Module): 파라미터를 가져올 소스 네트워크
        tau (float, 0 < τ < 1): 업데이트 비율 (가중치 계수)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    하드 업데이트: 소스 네트워크의 파라미터를 타겟에 그대로 복사
    초기화 시 타겟 네트워크를 소스와 동일하게 만들 때 사용

    인자:
        target (torch.nn.Module): 파라미터를 복사받을 타겟 네트워크
        source (torch.nn.Module): 파라미터를 제공할 소스 네트워크
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def average_gradients(model):
    """
    분산 학습 시 모든 워커의 그래디언트를 평균내는 함수
    각 워커가 독립적으로 계산한 그래디언트를 all_reduce로 합산 후 워커 수로 나눔

    인자:
        model (torch.nn.Module): 그래디언트를 평균낼 모델
    """
    size = float(dist.get_world_size())  # 전체 워커(프로세스) 수
    for param in model.parameters():
        # 모든 워커의 그래디언트를 합산 (in-place)
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size  # 워커 수로 나누어 평균 계산


def init_processes(rank, size, fn, backend='gloo'):
    """
    분산 학습 환경을 초기화하는 함수
    PyTorch의 분산 프로세스 그룹을 설정함

    인자:
        rank (int): 현재 프로세스의 순위 (0부터 시작)
        size (int): 전체 프로세스 수
        fn (function): 초기화 후 실행할 학습 함수
        backend (str): 통신 백엔드 (기본값: 'gloo', CPU용)
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 마스터 노드 주소 (로컬)
    os.environ['MASTER_PORT'] = '29500'  # 마스터 노드 포트
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)  # 분산 환경 설정 후 학습 함수 실행


def onehot_from_logits(logits, eps=0.0, dim=1):
    """
    로짓 값으로부터 원-핫 벡터를 생성하는 함수
    엡실론-탐욕(epsilon-greedy) 전략을 선택적으로 적용

    동작:
        1. 로짓에서 최대값을 가진 위치를 원-핫으로 변환
        2. eps > 0이면 일정 확률로 랜덤 행동을 선택

    인자:
        logits (Tensor): 배치 형태의 로짓 값 [batch_size, n_actions]
        eps (float): 랜덤 행동 선택 확률 (기본값: 0.0, 완전 탐욕)
        dim (int): 최대값을 구할 차원 (기본값: 1)
    반환:
        Tensor: 원-핫 인코딩된 행동 [batch_size, n_actions]
    """
    # 현재 정책에 따른 최적 행동을 원-핫으로 변환
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # 랜덤 행동을 원-핫으로 생성
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # 엡실론 확률에 따라 최적 행동 또는 랜덤 행동 선택
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """
    Gumbel(0, 1) 분포에서 샘플을 추출
    Gumbel-Softmax 기법의 기본 구성 요소

    공식: -log(-log(U)), U ~ Uniform(0, 1)

    인자:
        shape: 출력 텐서의 형태
        eps (float): 수치 안정성을 위한 작은 상수 (log(0) 방지)
        tens_type: 텐서 타입 (CPU/GPU에 따라 다름)
    반환:
        Tensor: Gumbel 분포 샘플
    """
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, dim=1):
    """
    Gumbel-Softmax 분포에서 샘플을 추출
    로짓에 Gumbel 노이즈를 더한 후 temperature로 나누어 소프트맥스 적용

    온도(temperature)가 낮을수록 원-핫에 가까운 출력,
    높을수록 균일 분포에 가까운 출력

    인자:
        logits (Tensor): 로짓 값 [batch_size, n_class]
        temperature (float): 온도 파라미터 (양수)
        dim (int): 소프트맥스를 적용할 차원
    반환:
        Tensor: Gumbel-Softmax 확률 분포 샘플
    """
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=dim)


def gumbel_softmax(logits, temperature=1.0, hard=False, dim=1):
    """
    Gumbel-Softmax 샘플링 (선택적으로 이산화)
    이산 행동을 미분 가능하게 샘플링하는 핵심 기법

    hard=True일 때 Straight-Through 추정기 사용:
    - 순전파: 원-핫 (이산) 출력
    - 역전파: 연속적인 소프트 샘플의 그래디언트 사용
    이를 통해 이산 행동에 대해서도 역전파 학습이 가능함

    인자:
        logits (Tensor): 정규화되지 않은 로그 확률 [batch_size, n_class]
        temperature (float): 온도 파라미터 (양수, 기본값: 1.0)
        hard (bool): True면 원-핫으로 이산화, 역전파는 소프트 샘플 기반
        dim (int): 소프트맥스 차원
    반환:
        Tensor: [batch_size, n_class] Gumbel-Softmax 샘플
                hard=True면 원-핫, 아니면 확률 분포
    """
    y = gumbel_softmax_sample(logits, temperature, dim=dim)
    if hard:
        # Straight-Through 추정기: argmax로 원-핫 변환하되, 그래디언트는 y를 통해 전달
        y_hard = onehot_from_logits(y, dim=dim)
        y = (y_hard - y).detach() + y  # 순전파: y_hard 값 사용, 역전파: y의 그래디언트 사용
    return y


def firmmax_sample(logits, temperature, dim=1):
    """
    Firmmax 샘플링: Gumbel-Softmax의 변형
    temperature가 0이면 일반 소프트맥스, 아니면 Gumbel 노이즈를 temperature로 스케일링

    Gumbel-Softmax와의 차이: 노이즈를 temperature로 나누는 것이 아니라
    로짓 전체를 temperature로 나눔 → 다른 분포 특성

    인자:
        logits (Tensor): 로짓 값
        temperature (float): 온도 파라미터 (0이면 결정적)
        dim (int): 소프트맥스 차원
    반환:
        Tensor: 샘플링된 확률 분포
    """
    if temperature == 0:
        return F.softmax(logits, dim=dim)
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)) / temperature
    return F.softmax(y, dim=dim)


def categorical_sample(probs, use_cuda=False):
    """
    범주형 분포에서 샘플링하여 정수 인덱스와 원-핫 벡터를 반환

    인자:
        probs (Tensor): 행동 확률 분포 [batch_size, n_actions]
        use_cuda (bool): GPU 텐서 사용 여부
    반환:
        int_acs (Tensor): 샘플링된 행동 인덱스 [batch_size, 1]
        acs (Tensor): 원-핫 인코딩된 행동 [batch_size, n_actions]
    """
    int_acs = torch.multinomial(probs, 1)  # 확률에 따라 하나의 행동 인덱스 샘플링
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    # 0으로 채운 텐서에 scatter로 해당 인덱스 위치에 1을 채워 원-핫 생성
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs


def disable_gradients(module):
    """
    모듈의 모든 파라미터에 대해 그래디언트 계산을 비활성화
    크리틱이 정책 손실의 역전파에 영향받지 않도록 할 때 사용

    인자:
        module (nn.Module): 그래디언트를 비활성화할 모듈
    """
    for p in module.parameters():
        p.requires_grad = False


def enable_gradients(module):
    """
    모듈의 모든 파라미터에 대해 그래디언트 계산을 다시 활성화

    인자:
        module (nn.Module): 그래디언트를 활성화할 모듈
    """
    for p in module.parameters():
        p.requires_grad = True


def sep_clip_grad_norm(parameters, max_norm, norm_type=2):
    """
    파라미터별 개별 그래디언트 클리핑
    일반적인 torch.nn.utils.clip_grad_norm과 달리
    전체 파라미터의 통합 노름이 아닌 각 파라미터별로 독립적으로 클리핑

    그래디언트 폭발(gradient explosion)을 방지하면서도
    각 파라미터의 상대적 크기를 보존함

    인자:
        parameters: 클리핑할 파라미터 이터러블
        max_norm (float): 최대 허용 그래디언트 노름
        norm_type (float): 노름 유형 (2 = L2 노름, inf = 최대값 노름)
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))  # 그래디언트가 있는 파라미터만 필터링
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    for p in parameters:
        if norm_type == float('inf'):
            # 무한대 노름: 그래디언트의 절대값 최대치 사용
            p_norm = p.grad.data.abs().max()
        else:
            # L2 (또는 다른 p-노름) 계산
            p_norm = p.grad.data.norm(norm_type)
        # 클리핑 계수 계산: max_norm / (현재 노름 + 작은 상수)
        clip_coef = max_norm / (p_norm + 1e-6)
        if clip_coef < 1:
            # 노름이 max_norm을 초과할 때만 스케일링 (축소)
            p.grad.data.mul_(clip_coef)
