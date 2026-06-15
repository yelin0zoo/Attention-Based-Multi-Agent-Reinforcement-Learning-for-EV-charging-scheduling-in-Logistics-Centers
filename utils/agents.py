"""
어텐션 에이전트 모듈
각 에이전트의 정책 네트워크와 타겟 정책 네트워크를 관리하는 클래스
MAAC에서 각 에이전트는 독립적인 정책을 가지며, 중앙 크리틱을 공유함
"""
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam  # Adam 옵티마이저
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits  # 유틸리티 함수들
from utils.policies import DiscretePolicy  # 이산 행동 공간용 정책 네트워크


class AttentionAgent(object):
    """
    어텐션 기반 에이전트 클래스
    각 에이전트는 행동을 선택하는 정책(policy)과
    학습 안정성을 위한 타겟 정책(target_policy)을 가짐

    구조:
        - policy: 현재 학습 중인 정책 네트워크 (행동 선택에 사용)
        - target_policy: 타겟 Q값 계산에 사용되는 느리게 업데이트되는 정책
        - policy_optimizer: 정책 네트워크 파라미터를 최적화하는 Adam 옵티마이저
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        에이전트 초기화

        인자:
            num_in_pol (int): 정책 네트워크 입력 차원 (관측 공간 크기)
            num_out_pol (int): 정책 네트워크 출력 차원 (행동 공간 크기)
            hidden_dim (int): 은닉층 뉴런 수 (기본값: 64)
            lr (float): 학습률 (기본값: 0.01)
            onehot_dim (int): 원-핫 인코딩 레이블 차원 (기본값: 0, 미사용)
        """
        # 현재 학습 중인 정책 네트워크 생성
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        # 타겟 정책 네트워크 생성 (동일한 구조)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        # 타겟 정책의 파라미터를 현재 정책과 동일하게 초기화 (하드 업데이트)
        hard_update(self.target_policy, self.policy)
        # 정책 네트워크 학습을 위한 Adam 옵티마이저 설정
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        주어진 관측값에 대해 행동을 선택하는 함수

        인자:
            obs (PyTorch Variable): 현재 에이전트의 관측값 배치
            explore (bool): True면 확률적으로 샘플링 (탐험),
                           False면 가장 확률이 높은 행동 선택 (활용)
        반환:
            action (PyTorch Variable): 선택된 행동 (원-핫 인코딩 형태)
        """
        return self.policy(obs, sample=explore)

    def get_params(self):
        """
        에이전트의 모든 학습 가능한 파라미터를 딕셔너리로 반환
        모델 저장 시 사용됨

        반환:
            dict: 정책, 타겟 정책, 옵티마이저의 state_dict를 포함하는 딕셔너리
        """
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        """
        저장된 파라미터를 에이전트에 로드하는 함수
        모델 복원 시 사용됨

        인자:
            params (dict): get_params()로 저장된 파라미터 딕셔너리
        """
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
