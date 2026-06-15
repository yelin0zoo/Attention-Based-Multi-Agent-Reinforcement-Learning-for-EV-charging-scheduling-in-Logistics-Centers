"""
AttentionSAC_CMDP: MAAC의 CMDP (Constrained MDP) 확장

Lagrangian Relaxation 방식으로 제약 조건을 학습에 통합:
  - 과부하 제약 C1: 그리드 과부하 비용 ≤ d1 (기본값 0)
  - 미충전 제약 C2: 미충전 비용    ≤ d2 (기존 성능 기준)

핵심 변경:
  1. 제약 크리틱(Q_c): 각 제약마다 별도의 AttentionCritic (엔트로피 항 없음)
  2. 라그랑지안 승수(λ): 제약 위반 시 λ 증가 → 정책 손실에 페널티 부가
  3. 정책 손실: L(π) = E[log π * (log π / α - A_r + Σ_k λ_k * A_c_k)]

기존 파일 수정 없이 AttentionSAC를 상속하여 구현.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam

from algorithms.attention_sac import AttentionSAC
from utils.critics import AttentionCritic
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients

MSELoss = torch.nn.MSELoss()


class AttentionSACCMDP(AttentionSAC):
    """
    CMDP 확장 AttentionSAC

    부모 클래스(AttentionSAC)의 모든 기능을 유지하면서
    제약 크리틱과 라그랑지안 승수를 추가.

    추가 구성 요소:
    - constraint_critics     : 제약별 AttentionCritic (n_constraints개)
    - target_constraint_critics : 제약별 타겟 크리틱
    - constraint_optimizers  : 제약 크리틱 옵티마이저
    - lambdas                : 라그랑지안 승수 [n_constraints] (torch.Tensor)
    """

    def __init__(self,
                 agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 # ── CMDP 전용 파라미터 ───────────────────────────────
                 n_constraints=2,
                 constraint_thresholds=None,   # [d1, d2] 제약 임계값
                 lambda_lr=1e-3,               # λ 업데이트 학습률
                 lambda_max=10.0,              # λ 클리핑 상한
                 **kwargs):
        """
        인자:
            n_constraints (int)           : 제약 수 (기본 2: 과부하, 미충전)
            constraint_thresholds (list)  : 각 제약의 상한 d_k
                None이면 모두 0.0 (위반 불허)
            lambda_lr (float)             : 라그랑지안 승수 학습률
            lambda_max (float)            : λ 최대값 (무한 발산 방지)
        """
        # 부모 클래스 초기화 (reward critic, 정책, 타겟 네트워크 포함)
        super().__init__(
            agent_init_params=agent_init_params,
            sa_size=sa_size,
            gamma=gamma, tau=tau,
            pi_lr=pi_lr, q_lr=q_lr,
            reward_scale=reward_scale,
            pol_hidden_dim=pol_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
            attend_heads=attend_heads,
            **kwargs
        )

        self.n_constraints = n_constraints

        # 제약 임계값: 각 d_k
        if constraint_thresholds is None:
            self.constraint_thresholds = [0.0] * n_constraints
        else:
            self.constraint_thresholds = list(constraint_thresholds)

        self.lambda_lr  = lambda_lr
        self.lambda_max = lambda_max

        # ── 라그랑지안 승수 초기화 (0으로 시작, 위반 시 증가) ──────────
        # requires_grad=False: λ는 closed-form 업데이트로 직접 수정
        self.lambdas = [torch.tensor(0.0, dtype=torch.float32)
                        for _ in range(n_constraints)]

        # ── 제약 크리틱 생성 ─────────────────────────────────────────
        self.constraint_critics = [
            AttentionCritic(sa_size,
                            hidden_dim=critic_hidden_dim,
                            attend_heads=attend_heads)
            for _ in range(n_constraints)
        ]
        self.target_constraint_critics = [
            AttentionCritic(sa_size,
                            hidden_dim=critic_hidden_dim,
                            attend_heads=attend_heads)
            for _ in range(n_constraints)
        ]
        # 타겟을 현재 크리틱과 동일하게 초기화
        for cc, tcc in zip(self.constraint_critics,
                           self.target_constraint_critics):
            hard_update(tcc, cc)

        # 제약 크리틱 옵티마이저
        self.constraint_optimizers = [
            Adam(cc.parameters(), lr=q_lr, weight_decay=1e-3)
            for cc in self.constraint_critics
        ]

        # 디바이스 추적 변수 (부모 변수 외 추가)
        self.constraint_critic_dev = 'cpu'
        self.trgt_constraint_critic_dev = 'cpu'

    # ─────────────────────────────────────────────────────────────────────
    # 제약 크리틱 업데이트
    # ─────────────────────────────────────────────────────────────────────
    def update_constraint_critics(self, sample, logger=None):
        """
        제약 크리틱 Q_c 업데이트 (벨만 방정식, 엔트로피 항 없음)

        타겟: Q_c(s,a) ← c_k + γ * Q_c_target(s', π_target(s'))

        비용(cost)은 정규화하지 않음 (0~1 or 명목 단위 유지).

        인자:
            sample (tuple): (obs, acs, rews, next_obs, dones, costs)
                costs: list of n_constraints × list of n_agents Tensors [N]
            logger: TensorBoard SummaryWriter
        반환:
            list of float: 제약별 크리틱 손실
        """
        obs, acs, rews, next_obs, dones, costs = sample

        # 타겟 정책으로 다음 행동 계산 (엔트로피 없음)
        with torch.no_grad():
            next_acs = [pi(ob, return_log_pi=False)
                        for pi, ob in zip(self.target_policies, next_obs)]
            trgt_cc_in = list(zip(next_obs, next_acs))

        cc_losses = []
        for k, (cc, tcc, cc_opt) in enumerate(zip(
                self.constraint_critics,
                self.target_constraint_critics,
                self.constraint_optimizers)):

            # 현재 상태-행동의 Q_c 계산
            cc_in = list(zip(obs, acs))
            cc_rets = cc(cc_in, regularize=True, logger=logger, niter=self.niter)

            # 타겟 Q_c (다음 상태, 타겟 정책 행동)
            with torch.no_grad():
                next_qcs = tcc(trgt_cc_in)  # list of [N,1] per agent

            qc_loss = 0.0
            for a_i, (pqc, regs), nqc in zip(
                    range(self.nagents), cc_rets, next_qcs):
                # 비용은 에이전트별로 저장: costs[k][a_i] → [N]
                cost_ai = costs[k][a_i].view(-1, 1)
                done_ai = dones[a_i].view(-1, 1)

                # 제약 TD 타겟 (엔트로피 없음)
                target_qc = cost_ai + self.gamma * nqc * (1 - done_ai)
                qc_loss += MSELoss(pqc, target_qc.detach())
                for reg in regs:
                    qc_loss += reg

            qc_loss.backward()
            cc.scale_shared_grads()
            torch.nn.utils.clip_grad_norm_(cc.parameters(), 10)
            cc_opt.step()
            cc_opt.zero_grad()

            if logger is not None:
                logger.add_scalar(f'losses/qc{k}_loss', qc_loss, self.niter)

            cc_losses.append(qc_loss.item())

        return cc_losses

    # ─────────────────────────────────────────────────────────────────────
    # 정책 업데이트 (부모 override)
    # ─────────────────────────────────────────────────────────────────────
    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        """
        CMDP 정책 업데이트

        손실 함수:
            L(π_i) = E[log π_i * (log π_i / α - A_r_i + Σ_k λ_k * A_c_k_i)]

        여기서:
            A_r_i  = Q_r(s,a) - V_r(s)         : 보상 어드밴티지
            A_c_k_i = Q_c_k(s,a) - V_c_k(s)   : 제약 k의 어드밴티지
            λ_k    : 제약 k의 라그랑지안 승수
        """
        obs, acs, rews, next_obs, dones, costs = sample

        samp_acs     = []
        all_probs    = []
        all_log_pis  = []
        all_pol_regs = []

        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            if logger is not None:
                logger.add_scalar(f'agent{a_i}/policy_entropy', ent, self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        # 보상 크리틱: Q_r 및 모든 행동의 Q_r 값
        critic_in   = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)

        # 제약 크리틱: Q_c_k 및 모든 행동의 Q_c_k 값
        cc_rets_all = []
        for cc in self.constraint_critics:
            cc_rets_all.append(cc(critic_in, return_all_q=True))

        pol_losses = []
        for a_i, probs, log_pi, pol_regs, (q_r, all_q_r) in zip(
                range(self.nagents), all_probs, all_log_pis,
                all_pol_regs, critic_rets):

            curr_agent = self.agents[a_i]

            # 보상 어드밴티지
            v_r     = (all_q_r * probs).sum(dim=1, keepdim=True)
            pol_target_r = q_r - v_r          # [N, 1]

            # 제약 어드밴티지 합산: Σ_k λ_k * A_c_k
            constraint_penalty = torch.zeros_like(pol_target_r)
            for k, cc_rets in enumerate(cc_rets_all):
                q_c, all_q_c = cc_rets[a_i]  # [N,1], [N, n_actions]
                v_c = (all_q_c * probs).sum(dim=1, keepdim=True)
                pol_target_c = q_c - v_c      # [N, 1]
                lam = self.lambdas[k].to(pol_target_c.device)
                constraint_penalty = constraint_penalty + lam * pol_target_c

            # CMDP 정책 손실
            if soft:
                combined_target = (log_pi / self.reward_scale
                                   - pol_target_r
                                   + constraint_penalty)
                pol_loss = (log_pi * combined_target.detach()).mean()
            else:
                combined_target = -pol_target_r + constraint_penalty
                pol_loss = (log_pi * combined_target.detach()).mean()

            for reg in pol_regs:
                pol_loss += 1e-3 * reg

            # 제약 크리틱의 그래디언트 비활성화 후 역전파
            disable_gradients(self.critic)
            for cc in self.constraint_critics:
                disable_gradients(cc)

            pol_loss.backward()

            enable_gradients(self.critic)
            for cc in self.constraint_critics:
                enable_gradients(cc)

            torch.nn.utils.clip_grad_norm_(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar(f'agent{a_i}/losses/pol_loss',
                                  pol_loss, self.niter)
            pol_losses.append(pol_loss.item())

        return pol_losses

    # ─────────────────────────────────────────────────────────────────────
    # 라그랑지안 승수 업데이트
    # ─────────────────────────────────────────────────────────────────────
    def update_lambdas(self, costs_batch):
        """
        λ_k = clip(λ_k + lr_λ * (mean_cost_k - d_k), 0, lambda_max)

        제약 위반(mean_cost > threshold)이면 λ 증가,
        만족(mean_cost < threshold)이면 λ 감소 (최소 0).

        인자:
            costs_batch: list of n_constraints × list of n_agents Tensors [N]
                샘플 미니배치에서 가져온 비용 값
        반환:
            list of float: 업데이트된 λ 값
        """
        updated = []
        for k in range(self.n_constraints):
            # 모든 에이전트의 평균 비용 계산
            mean_cost = torch.stack(
                [costs_batch[k][i].mean() for i in range(self.nagents)]
            ).mean().item()

            d_k   = self.constraint_thresholds[k]
            lam_k = self.lambdas[k].item()

            # 이중 경사하강법 업데이트 (closed-form)
            new_lam = lam_k + self.lambda_lr * (mean_cost - d_k)
            new_lam = max(0.0, min(self.lambda_max, new_lam))
            self.lambdas[k] = torch.tensor(new_lam, dtype=torch.float32)
            updated.append(new_lam)

        return updated

    # ─────────────────────────────────────────────────────────────────────
    # 타겟 네트워크 업데이트 (부모 override)
    # ─────────────────────────────────────────────────────────────────────
    def update_all_targets(self):
        """보상 크리틱 + 제약 크리틱 + 정책 타겟 네트워크 소프트 업데이트"""
        super().update_all_targets()  # 보상 크리틱 & 정책 타겟 업데이트
        for cc, tcc in zip(self.constraint_critics,
                           self.target_constraint_critics):
            soft_update(tcc, cc, self.tau)

    # ─────────────────────────────────────────────────────────────────────
    # 디바이스 관리 (부모 override)
    # ─────────────────────────────────────────────────────────────────────
    def prep_training(self, device='gpu'):
        """제약 크리틱도 학습 모드로 전환 및 지정 디바이스로 이동"""
        super().prep_training(device=device)

        fn = (lambda x: x.cuda()) if device == 'gpu' else (lambda x: x.cpu())

        for cc in self.constraint_critics:
            cc.train()
        for tcc in self.target_constraint_critics:
            tcc.train()

        if not self.constraint_critic_dev == device:
            self.constraint_critics = [fn(cc) for cc in self.constraint_critics]
            self.constraint_critic_dev = device
        if not self.trgt_constraint_critic_dev == device:
            self.target_constraint_critics = [
                fn(tcc) for tcc in self.target_constraint_critics
            ]
            self.trgt_constraint_critic_dev = device

        # λ도 동일 디바이스로 이동
        dev = torch.device('cuda' if device == 'gpu' else 'cpu')
        self.lambdas = [lam.to(dev) for lam in self.lambdas]

    def prep_rollouts(self, device='cpu'):
        """정책만 평가 모드 (크리틱은 유지) - 부모와 동일하지만 λ 디바이스 동기화"""
        super().prep_rollouts(device=device)
        dev = torch.device('cuda' if device == 'gpu' else 'cpu')
        self.lambdas = [lam.to(dev) for lam in self.lambdas]

    # ─────────────────────────────────────────────────────────────────────
    # 저장 / 복원 (부모 override)
    # ─────────────────────────────────────────────────────────────────────
    def save(self, filename):
        """
        보상 크리틱, 제약 크리틱, 정책, λ, 설정을 하나의 파일에 저장.

        저장 키:
            init_dict          : 모델 재생성 파라미터
            agent_params       : 에이전트별 정책 파라미터
            critic_params      : 보상 크리틱 파라미터 (부모와 동일)
            constraint_params  : 제약 크리틱 파라미터 (리스트)
            lambdas            : 현재 λ 값 (리스트)
        """
        self.prep_training(device='cpu')  # 저장 전 CPU로 이동
        save_dict = {
            'init_dict': self.init_dict,
            'agent_params': [a.get_params() for a in self.agents],
            'critic_params': {
                'critic':           self.critic.state_dict(),
                'target_critic':    self.target_critic.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
            },
            'constraint_params': [
                {
                    'critic':           cc.state_dict(),
                    'target_critic':    tcc.state_dict(),
                    'optimizer':        opt.state_dict(),
                }
                for cc, tcc, opt in zip(
                    self.constraint_critics,
                    self.target_constraint_critics,
                    self.constraint_optimizers)
            ],
            'lambdas': [lam.item() for lam in self.lambdas],
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env,
                      gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128,
                      critic_hidden_dim=128,
                      attend_heads=4,
                      n_constraints=2,
                      constraint_thresholds=None,
                      lambda_lr=1e-3,
                      lambda_max=10.0,
                      **kwargs):
        """환경에서 CMDP 모델 자동 초기화"""
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space, env.observation_space):
            agent_init_params.append({
                'num_in_pol':  obsp.shape[0],
                'num_out_pol': acsp.n,
            })
            sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {
            'gamma': gamma, 'tau': tau,
            'pi_lr': pi_lr, 'q_lr': q_lr,
            'reward_scale': reward_scale,
            'pol_hidden_dim': pol_hidden_dim,
            'critic_hidden_dim': critic_hidden_dim,
            'attend_heads': attend_heads,
            'agent_init_params': agent_init_params,
            'sa_size': sa_size,
            'n_constraints': n_constraints,
            'constraint_thresholds': constraint_thresholds,
            'lambda_lr': lambda_lr,
            'lambda_max': lambda_max,
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        저장된 파일에서 CMDP 모델 복원.

        인자:
            filename   : .pt 파일 경로
            load_critic: 크리틱(보상+제약) 파라미터도 로드할지 여부
        """
        save_dict = torch.load(filename, map_location='cpu')
        instance  = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']

        # 에이전트 정책 복원
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            # 보상 크리틱 복원
            cp = save_dict['critic_params']
            instance.critic.load_state_dict(cp['critic'])
            instance.target_critic.load_state_dict(cp['target_critic'])
            instance.critic_optimizer.load_state_dict(cp['critic_optimizer'])

            # 제약 크리틱 복원
            if 'constraint_params' in save_dict:
                for k, ccp in enumerate(save_dict['constraint_params']):
                    instance.constraint_critics[k].load_state_dict(ccp['critic'])
                    instance.target_constraint_critics[k].load_state_dict(
                        ccp['target_critic'])
                    instance.constraint_optimizers[k].load_state_dict(
                        ccp['optimizer'])

        # λ 복원
        if 'lambdas' in save_dict:
            instance.lambdas = [
                torch.tensor(float(v), dtype=torch.float32)
                for v in save_dict['lambdas']
            ]

        return instance
