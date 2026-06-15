"""
정책 네트워크 모듈
에이전트가 관측값을 입력받아 행동을 출력하는 신경망 정의
기본 정책(BasePolicy)과 이산 행동 공간용 정책(DiscretePolicy)을 포함
"""
import torch
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 활성화 함수 등 신경망 연산
from utils.misc import onehot_from_logits, categorical_sample  # 행동 샘플링 유틸리티


class BasePolicy(nn.Module):
    """
    기본 정책 네트워크
    3층 완전연결(FC) 신경망으로 구성됨

    구조: 입력 → [배치정규화] → FC1 → LeakyReLU → FC2 → LeakyReLU → FC3 → 출력

    선택적으로 입력에 배치 정규화(BatchNorm)를 적용하여 학습 안정성을 높이고,
    원-핫 레이블을 추가 입력으로 받을 수 있음 (에이전트 식별 등)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        인자:
            input_dim (int): 입력 차원 (관측 공간 크기)
            out_dim (int): 출력 차원 (행동 공간 크기)
            hidden_dim (int): 은닉층 뉴런 수 (기본값: 64)
            nonlin (function): 은닉층에 적용할 비선형 활성화 함수 (기본값: LeakyReLU)
            norm_in (bool): 입력에 배치 정규화를 적용할지 여부 (기본값: True)
            onehot_dim (int): 원-핫 인코딩 입력의 추가 차원 (기본값: 0)
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # 입력 정규화 활성화 시
            # 배치 정규화 적용 (affine=False: 학습 가능한 스케일/시프트 파라미터 없음)
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            # 정규화 미적용: 항등 함수 (입력을 그대로 통과)
            self.in_fn = lambda x: x
        # 3층 완전연결 네트워크 정의
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)  # 입력층 → 은닉층1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 은닉층1 → 은닉층2
        self.fc3 = nn.Linear(hidden_dim, out_dim)  # 은닉층2 → 출력층
        self.nonlin = nonlin  # 비선형 활성화 함수 저장

    def forward(self, X):
        """
        순전파(forward pass): 관측값을 입력받아 행동 로짓(logit)을 출력

        인자:
            X (PyTorch Matrix): 관측값 배치
                또는 (관측값, 원-핫 레이블) 튜플
        반환:
            out (PyTorch Matrix): 행동 로짓 (소프트맥스 적용 전 원시 출력값)
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X  # 튜플인 경우 관측값과 원-핫 레이블 분리
        inp = self.in_fn(X)  # 배치 정규화 적용 (원-핫은 정규화하지 않음)
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)  # 원-핫 레이블을 입력에 연결(concatenate)
        h1 = self.nonlin(self.fc1(inp))  # 첫 번째 은닉층 + 활성화
        h2 = self.nonlin(self.fc2(h1))  # 두 번째 은닉층 + 활성화
        out = self.fc3(h2)  # 출력층 (활성화 없음)
        return out


class DiscretePolicy(BasePolicy):
    """
    이산 행동 공간용 정책 네트워크
    BasePolicy를 상속하며, 소프트맥스를 통해 행동 확률 분포를 생성하고
    범주형 샘플링으로 이산 행동을 선택함

    SAC(Soft Actor-Critic)에 필요한 로그 확률, 엔트로피 등도 반환 가능
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        """
        관측값을 입력받아 행동과 부가 정보를 반환

        인자:
            obs: 관측값 배치
            sample (bool): True면 확률 분포에서 샘플링, False면 argmax (결정적 선택)
            return_all_probs (bool): True면 모든 행동의 확률 분포도 반환
            return_log_pi (bool): True면 선택된 행동의 로그 확률도 반환 (SAC 학습에 필요)
            regularize (bool): True면 정규화 항도 반환 (출력 크기 패널티)
            return_entropy (bool): True면 정책 엔트로피도 반환 (탐험 지표)
        반환:
            단일 반환: 선택된 행동 (원-핫 인코딩)
            다중 반환: [행동, 확률분포, 로그확률, 정규화항, 엔트로피] 중 요청된 것들의 리스트
        """
        out = super(DiscretePolicy, self).forward(obs)  # 기본 네트워크의 순전파로 로짓 계산
        probs = F.softmax(out, dim=1)  # 로짓을 소프트맥스로 확률 분포로 변환
        on_gpu = next(self.parameters()).is_cuda  # 모델이 GPU에 있는지 확인
        if sample:
            # 범주형 분포에서 확률적으로 행동 샘플링 (탐험)
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            # 가장 높은 확률의 행동을 원-핫으로 변환 (결정적 선택)
            act = onehot_from_logits(probs)
        rets = [act]  # 반환값 리스트 (기본: 행동)
        if return_log_pi or return_entropy:
            # 로그 확률 계산 (수치 안정성을 위해 log_softmax 사용)
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)  # 전체 확률 분포 추가
        if return_log_pi:
            # 선택된 행동의 로그 확률 추출 (gather로 해당 인덱스의 값만 선택)
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            # 출력 로짓의 제곱 평균을 정규화 항으로 추가 (너무 큰 값 방지)
            rets.append([(out**2).mean()])
        if return_entropy:
            # 정책 엔트로피 계산: H(π) = -Σ π(a) log π(a)
            # 엔트로피가 높을수록 정책이 더 균일하게 탐험함
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]  # 행동만 반환
        return rets  # 요청된 모든 값을 리스트로 반환
