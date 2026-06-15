"""
AttentionSAC: MAAC의 핵심 메커니즘인 'Attention'과 'SAC'의 결합을 강조한 코드상의 구현 클래스명
멀티에이전트 환경에서 어텐션 기반 중앙 크리틱(공유)과 개별 정책 네트워크(분산)를 학습

중앙 크리틱에 '어텐션(Attention) 메커니즘'을 결합
'Attention 메커니즘': 어떤 에이전트의 정보가 현재 상황에서 중요한지 동적으로 가중치를 계산하여 정보를 통합

구조:
- 각 에이전트: 독립적인 정책 네트워크 (분산 실행)
- 중앙 크리틱: 어텐션 메커니즘으로 다른 에이전트 정보 통합 (중앙 학습)
- CTDE (Centralized Training, Decentralized Execution) 패러다임 구현

학습 과정:
1. 크리틱 업데이트: TD 오차 + 소프트 벨만 방정식으로 Q함수 학습
2. 정책 업데이트: 크리틱의 Q값을 사용하여 정책 개선 (SAC 목표)
3. 타겟 네트워크: 소프트 업데이트로 학습 안정성 확보
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam  # Adam 옵티마이저 (신경망의 가중치(파라미터)를 업데이트할 때 쓰임)
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent  # 개별 에이전트 클래스
from utils.critics import AttentionCritic  # 어텐션 크리틱 네트워크

MSELoss = torch.nn.MSELoss()  # 평균 제곱 오차 손실 함수 (크리틱 학습용)


class AttentionSAC(object):
    """
    어텐션 기반 멀티에이전트 SAC (Soft Actor-Critic) Wrapper 클래스

    핵심 특징:
    1. 중앙 크리틱 + 분산 정책 (CTDE)
    2. 멀티헤드 어텐션으로 에이전트 간 동적 정보 통합
    3. SAC의 최대 엔트로피 프레임워크로 탐험-활용 균형 자동 조절
    4. 타겟 네트워크를 통한 안정적인 학습

    구성 요소:
    - agents: 각 에이전트의 정책 및 타겟 정책
    - critic / target_critic: 공유 어텐션 크리틱 네트워크
    - critic_optimizer: 크리틱 학습 옵티마이저
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        인자:
            agent_init_params (list of dict): 각 에이전트 초기화 파라미터
                - num_in_pol (int): 정책 입력 차원 (관측 공간)
                - num_out_pol (int): 정책 출력 차원 (행동 공간)
            sa_size (list of (int, int)): 각 에이전트의 (상태 차원, 행동 차원) 리스트
            gamma (float): 할인 계수 (미래 보상의 현재 가치 감쇠, 기본값: 0.95)
            tau (float): 타겟 네트워크 소프트 업데이트 비율 (기본값: 0.01)
            pi_lr (float): 정책 학습률 (기본값: 0.01)
            q_lr (float): 크리틱 학습률 (기본값: 0.01)
            reward_scale (float): 보상 스케일링 계수 (기본값: 10.0) - 온도 파라미터
                최적 정책의 엔트로피에 영향 - 클수록 낮은 엔트로피(결정적) 선호
            pol_hidden_dim (int): 정책 네트워크 은닉층 차원 (기본값: 128)
            critic_hidden_dim (int): 크리틱 네트워크 은닉층 차원 (기본값: 128)
            attend_heads (int): 어텐션 헤드 수 (기본값: 4)
        """
        self.nagents = len(sa_size)  # 에이전트 수

        # 각 에이전트별 정책 네트워크 생성
        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
        # 중앙 어텐션 크리틱 네트워크 생성
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        # 타겟 크리틱 네트워크 생성 (동일 구조, 느리게 업데이트)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        # 타겟 크리틱을 현재 크리틱과 동일하게 초기화
        hard_update(self.target_critic, self.critic)
        # 크리틱 옵티마이저 (가중치 감쇄(weight decay) 적용으로 과적합 방지)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        # 하이퍼파라미터 저장
        self.agent_init_params = agent_init_params
        self.gamma = gamma  # 할인 계수
        self.tau = tau  # 타겟 네트워크 업데이트 비율
        self.pi_lr = pi_lr  # 정책 학습률
        self.q_lr = q_lr  # 크리틱 학습률
        self.reward_scale = reward_scale  # 보상 스케일
        # 디바이스 추적 변수 (불필요한 디바이스 전환 방지)
        self.pol_dev = 'cpu'  # 정책 네트워크 디바이스
        self.critic_dev = 'cpu'  # 크리틱 디바이스
        self.trgt_pol_dev = 'cpu'  # 타겟 정책 디바이스
        self.trgt_critic_dev = 'cpu'  # 타겟 크리틱 디바이스
        self.niter = 0  # 학습 반복 카운터 (로깅용)

    @property
    def policies(self):
        """모든 에이전트의 현재 정책 리스트 반환"""
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        """모든 에이전트의 타겟 정책 리스트 반환"""
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        모든 에이전트가 동시에 행동을 선택 (환경 상호작용용)

        인자:
            observations (list): 각 에이전트의 관측값 리스트
            explore (bool): True면 확률적 행동 (탐험), False면 결정적 행동 (활용)
        반환:
            list: 각 에이전트의 행동 텐서 리스트
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        중앙 크리틱 네트워크 업데이트

        소프트 벨만 방정식 기반 TD(Temporal Difference) 학습:
        Q(s,a) ← r + γ * (Q_target(s', a') - α * log π(a'|s'))

        여기서:
        - r: 현재 보상
        - γ: 할인 계수
        - Q_target: 타겟 크리틱으로 계산한 다음 상태의 Q값
        - α = 1/reward_scale: 온도 파라미터 (엔트로피 가중치)
        - log π: 다음 행동의 로그 확률 (소프트 벨만의 엔트로피 보너스)

        인자:
            sample (tuple): (관측, 행동, 보상, 다음관측, 종료) 미니배치
            soft (bool): True면 SAC 엔트로피 항 포함 (기본값: True)
            logger (SummaryWriter): TensorBoard 로거
        """
        obs, acs, rews, next_obs, dones = sample

        # === 타겟 Q값 계산 ===
        # 타겟 정책으로 다음 상태의 행동 및 로그 확률 계산
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)

        # 타겟 크리틱과 현재 크리틱의 입력 구성 (상태-행동 쌍)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        # 타겟 크리틱으로 다음 상태의 Q값 계산
        next_qs = self.target_critic(trgt_critic_in)
        # 현재 크리틱으로 Q값 + 정규화 항 계산
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)

        # === 크리틱 손실 loss 계산 ===
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            # 타겟 Q값 = 보상 + 할인계수 × 다음Q × (1 - 종료)
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                # SAC: 타겟에서 엔트로피 보너스 차감
                # 높은 엔트로피(낮은 로그확률)를 가진 정책에 보상 부여
                target_q -= log_pi / self.reward_scale
            # MSE 손실: |현재Q - 타겟Q|² (타겟은 그래디언트 차단)
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # 어텐션 정규화 항 추가

        # === 역전파 및 파라미터 업데이트 ===
        q_loss.backward()  # 그래디언트 계산
        self.critic.scale_shared_grads()  # 공유 파라미터 그래디언트를 에이전트 수로 스케일링
        # 전체 그래디언트 Norm Clipping
        # Norm Clipping: Gradient Exploding를 막는 장치
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10)
        self.critic_optimizer.step()  # 파라미터 업데이트
        self.critic_optimizer.zero_grad()  # 그래디언트 초기화

        # TensorBoard에 손실 및 그래디언트 Norm 기록
        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1  # 반복 카운터 증가
        return q_loss.item()

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        """
        모든 에이전트의 정책 네트워크 업데이트
        "보상과 탐험 사이의 관계"를 조율

        SAC 정책 손실:
        L(π) = E[log π(a|s) * (log π(a|s) / α - Q(s,a) + V(s))]

        여기서:
        - log π(a|s): 선택된 행동의 로그 확률
        - α = reward_scale: 온도 파라미터
        - Q(s,a): 크리틱이 추정한 행동 가치
        - V(s) = Σ π(a|s) Q(s,a): 상태 가치 (모든 행동의 가중 평균 Q)

        핵심: 높은 Q값의 행동 확률을 높이면서, 적절한 엔트로피를 유지

        인자:
            sample (tuple): (관측, 행동, 보상, 다음관측, 종료) 미니배치
            soft (bool): True면 SAC 엔트로피 정규화 적용
            logger (SummaryWriter): TensorBoard 로거
        """
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []  # 현재 정책에서 새로 샘플링한 행동
        all_probs = []  # 전체 행동 확률 분포
        all_log_pis = []  # 선택된 행동의 로그 확률
        all_pol_regs = []  # 정책 정규화 항
        pol_losses = []  # 에이전트별 정책 loss 저장

        # 각 에이전트의 현재 정책에서 행동, 확률, 로그확률, 엔트로피 수집
        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            # 정책 엔트로피를 TensorBoard에 기록 (탐험 정도 모니터링)
            logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
                              self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        # 크리틱에 현재 관측과 새로 샘플링한 행동을 입력하여 Q값 계산
        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)  # 모든 행동의 Q값 반환

        # === 에이전트별 정책 업데이트 ===
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            # 상태 가치 V(s) = Σ π(a) * Q(s,a) (모든 행동에 대한 기대 Q값)
            v = (all_q * probs).sum(dim=1, keepdim=True)
            # 어드밴티지: A(s,a) = Q(s,a) - V(s) (선택된 행동이 평균 대비 얼마나 좋은지)
            pol_target = q - v
            if soft:
                # SAC 정책 손실: log π * (log π / α - A) → 엔트로피 최대화 + Q 최대화
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                # 일반 정책 그래디언트: log π * (-A)
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # 정책 출력 크기 정규화

            # 크리틱의 그래디언트를 비활성화하여 정책 손실이 크리틱을 업데이트하지 않도록 함
            disable_gradients(self.critic)
            pol_loss.backward()  # 정책 그래디언트 계산
            enable_gradients(self.critic)  # 크리틱 그래디언트 다시 활성화

            # 정책 그래디언트 Norm 클리핑 (안정적 학습)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()  # 정책 파라미터 업데이트
            curr_agent.policy_optimizer.zero_grad()  # 그래디언트 초기화

            # TensorBoard에 정책 손실 및 그래디언트 Norm 기록
            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)
            pol_losses.append(pol_loss.item())
        return pol_losses


    def update_all_targets(self):
        """
        모든 타겟 네트워크를 소프트 업데이트

        크리틱과 정책 모두의 타겟을 현재 네트워크 방향으로 천천히 이동:
        θ_target = (1 - τ) * θ_target + τ * θ_current

        이를 통해 학습의 안정성을 확보하고, 부트스트래핑의 분산을 줄임
        """
        soft_update(self.target_critic, self.critic, self.tau)  # 크리틱 타겟 업데이트
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)  # 각 에이전트의 정책 타겟 업데이트

    def prep_training(self, device='gpu'):
        """
        학습 모드 설정: 모든 네트워크를 학습(train) 모드로 전환하고
        필요한 경우 지정된 디바이스(GPU/CPU)로 이동

        본격적인 업데이트(역전파)를 위해 모든 신경망을
        '학습(Train) 모드'로 켜고, 고속 연산을 위해 모델을 GPU로 배치하는 준비 작업

        인자:
            device (str): 'gpu' 또는 'cpu'
        """
        self.critic.train()  # 크리틱 학습 모드
        self.target_critic.train()  # 타겟 크리틱도 학습 모드 (BatchNorm 통계 업데이트용)
        for a in self.agents:
            a.policy.train()  # 정책 학습 모드
            a.target_policy.train()  # 타겟 정책 학습 모드
        # 디바이스 전환 함수 설정
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # 현재 디바이스와 다른 경우에만 전환 (불필요한 전환 방지)
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        """
        롤아웃(데이터 수집) 모드 설정
        정책 네트워크만 평가(eval) 모드로 전환하고 CPU로 이동

        평가 모드에서는 BatchNorm이 이동 평균을 사용하고, Dropout이 비활성화됨
        롤아웃에는 정책만 필요하므로 크리틱은 전환하지 않음
    

        인자:
            device (str): 'gpu' 또는 'cpu' (보통 'cpu' 사용)
        """
        for a in self.agents:
            a.policy.eval()  # 정책을 평가 모드로 전환
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # 롤아웃에는 정책 네트워크만 필요
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        모든 에이전트의 학습된 파라미터를 하나의 파일에 저장

        저장 내용:
        - init_dict: 모델 재생성에 필요한 초기화 파라미터
        - agent_params: 각 에이전트의 정책/타겟정책/옵티마이저 상태
        - critic_params: 크리틱/타겟크리틱/크리틱옵티마이저 상태

        인자:
            filename (str/Path): 저장할 파일 경로 (.pt 확장자)
        """
        self.prep_training(device='cpu')  # 저장 전 모든 파라미터를 CPU로 이동
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        멀티에이전트 환경으로부터 자동으로 모델을 초기화하는 클래스 메서드
        환경의 관측/행동 공간을 분석하여 적절한 네트워크 구조를 생성

        인자:
            env: 멀티에이전트 Gym 환경
            gamma (float): 할인 계수
            tau (float): 타겟 네트워크 업데이트 비율
            pi_lr (float): 정책 학습률
            q_lr (float): 크리틱 학습률
            reward_scale (float): 보상 스케일링
            pol_hidden_dim (int): 정책 은닉층 차원
            critic_hidden_dim (int): 크리틱 은닉층 차원
            attend_heads (int): 어텐션 헤드 수
        반환:
            AttentionSAC: 초기화된 모델 인스턴스
        """
        agent_init_params = []
        sa_size = []
        # 각 에이전트의 행동/관측 공간에서 네트워크 차원 추출
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0],  # 관측 차원 → 정책 입력
                                      'num_out_pol': acsp.n})  # 행동 수 → 정책 출력
            sa_size.append((obsp.shape[0], acsp.n))  # (상태 차원, 행동 차원)

        # 초기화 파라미터를 딕셔너리로 정리 (저장/복원에 사용)
        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)  # 인스턴스 생성
        instance.init_dict = init_dict  # 복원용 초기화 딕셔너리 저장
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        저장된 파일로부터 모델을 복원하는 클래스 메서드

        인자:
            filename (str): 저장된 모델 파일 경로
            load_critic (bool): 크리틱 파라미터도 로드할지 여부
                학습 재개 시 True, 실행만 할 때는 False
        반환:
            AttentionSAC: 복원된 모델 인스턴스
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])  # 저장된 초기화 파라미터로 새 인스턴스 생성
        instance.init_dict = save_dict['init_dict']
        # 각 에이전트의 정책 파라미터 로드
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            # 크리틱 파라미터도 로드 (학습 재개 시 필요)
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance
