"""
train_sac.py  —  Single-Agent Discrete SAC  (EV 충전 대조군)

기반 논문 : Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
기반 코드 : Atari-SAC-Discrete (rltorch)

논문 파일 → 본 파일 변환 내역
────────────────────────────────────────────────────────────────
rltorch/policy/categorical.py        → CategoricalPolicy
  · ConvCategoricalPolicy            → Conv backbone 제거, MLP 교체
  · 출력 헤드 1개                    → 충전기 n_agents개 독립 헤드

rltorch/q_function/discrete.py       → TwinedDiscreteQNetwork
  · TwinedDiscreteConvQNetwork       → Conv 제거, MLP 교체
  · 출력 헤드 1개 (action_n)         → 충전기 n_agents개 독립 헤드

rltorch/memory/base.py               → Memory
  · actions shape (1,)               → (n_agents,) 정수 배열

rltorch/agent/sac_discrete/base.py   → SacDiscreteAgent (일부)
  · calc_current_q(), calc_target_q()
  · explore(), exploit()

rltorch/agent/sac_discrete/learner.py → SacDiscreteAgent (일부)
  · calc_critic_loss(), calc_policy_loss(), calc_entropy_loss()
  · learn()  ← Actor/Learner 단일 클래스로 통합

rltorch/agent/sac_discrete/actor.py  → run()
  · act_episode()                    → 단일 프로세스 학습 루프

rltorch/agent/utils.py               → update_params() 유지
────────────────────────────────────────────────────────────────
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from pathlib import Path
from tensorboardX import SummaryWriter

from utils.make_env import make_env
from utils.misc import soft_update, hard_update
from utils.single_wrapper import MultiToSingleWrapper


# ─────────────────────────────────────────────────────────────
# 유틸  [출처: rltorch/agent/utils.py  update_params()]
# ─────────────────────────────────────────────────────────────

def update_params(optim, loss, grad_clip, network=None, retain_graph=False):
    """
    논문 utils.py의 update_params() 그대로.
    zero_grad → backward → clip → step 순서 보장.
    """
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None and network is not None:
        nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optim.step()


# ─────────────────────────────────────────────────────────────
# 1. Policy  [출처: rltorch/policy/categorical.py]
# ─────────────────────────────────────────────────────────────

class CategoricalPolicy(nn.Module):
    """
    논문: ConvCategoricalPolicy
    변경: Conv backbone → MLP,  출력 헤드 1개 → n_agents개

    논문의 sample() 반환 형태를 그대로 유지:
        (actions, action_probs, log_action_probs, greedy_actions)
    """

    def __init__(self, obs_dim, action_n, n_agents, hidden_dim=256):
        super().__init__()
        self.n_agents = n_agents
        self.action_n = action_n

        # 논문의 Conv backbone → MLP로 교체
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 논문의 단일 출력 헤드 → 충전기 수만큼 독립 헤드
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_n) for _ in range(n_agents)
        ])

    def forward(self, states):
        """list[n_agents] of (B, action_n) 확률 반환"""
        h = self.net(states)
        return [F.softmax(head(h), dim=1) for head in self.heads]

    def sample(self, states):
        """
        논문 categorical.py  L26~35  sample() 그대로.

        논문:
            action_probs = self.policy(state)
            greedy_actions = torch.argmax(action_probs, dim=1, keepdim=True)
            categorical = Categorical(action_probs)
            actions = categorical.sample().view(-1, 1)
            log_action_probs = torch.log(action_probs + (action_probs==0)*1e-8)
            return actions, action_probs, log_action_probs, greedy_actions

        반환:
            actions        (B, n_agents)  정수 — 환경에 바로 전달
            action_probs   list[n_agents] of (B, action_n)
            log_action_probs list[n_agents] of (B, action_n)
            greedy_actions (B, n_agents)  정수
        """
        h = self.net(states)

        actions_list, probs_list, log_probs_list, greedy_list = [], [], [], []

        for head in self.heads:
            probs = F.softmax(head(h), dim=1)

            # 논문 L32~33: log(probs + eps)  — log(0) 방지
            log_probs = torch.log(probs + (probs == 0.0).float() * 1e-8)

            # 논문 L29~30: Categorical 샘플링
            categorical     = Categorical(probs)
            actions         = categorical.sample().view(-1, 1)          # (B,1)
            greedy_actions  = torch.argmax(probs, dim=1, keepdim=True)  # (B,1)

            actions_list.append(actions)
            probs_list.append(probs)
            log_probs_list.append(log_probs)
            greedy_list.append(greedy_actions)

        return (
            torch.cat(actions_list, dim=1),    # (B, n_agents)
            probs_list,                         # list[n_agents] of (B, action_n)
            log_probs_list,                     # list[n_agents] of (B, action_n)
            torch.cat(greedy_list,  dim=1),    # (B, n_agents)
        )


# ─────────────────────────────────────────────────────────────
# 2. Q-Network  [출처: rltorch/q_function/discrete.py]
# ─────────────────────────────────────────────────────────────

class DiscreteQNetwork(nn.Module):
    """
    논문: DiscreteConvQNetwork
    변경: Conv + Dueling → MLP,  단일 헤드 → n_agents개 독립 헤드

    논문의 핵심 개념 유지:
        Q(s) → 모든 행동의 Q값 동시 출력  (이산 SAC 핵심)
        연속 SAC처럼 Q(s,a)를 입력받지 않음
    """

    def __init__(self, obs_dim, action_n, n_agents, hidden_dim=256):
        super().__init__()

        # 논문의 Conv base → MLP로 교체
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 논문의 단일 헤드 (V+A Dueling) → MLP 독립 헤드 n_agents개
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_n) for _ in range(n_agents)
        ])

    def forward(self, states):
        """list[n_agents] of (B, action_n) Q값 반환"""
        h = self.net(states)
        return [head(h) for head in self.heads]


class TwinedDiscreteQNetwork(nn.Module):
    """
    논문: TwinedDiscreteConvQNetwork  (discrete.py  L23~35)
    변경: 내부 Q망을 DiscreteQNetwork로 교체

    논문과 동일하게 .Q1 / .Q2 속성 유지
    → learner.py처럼 Q1, Q2 각각 독립 optimizer 적용 가능
    """

    def __init__(self, obs_dim, action_n, n_agents, hidden_dim=256):
        super().__init__()
        self.Q1 = DiscreteQNetwork(obs_dim, action_n, n_agents, hidden_dim)
        self.Q2 = DiscreteQNetwork(obs_dim, action_n, n_agents, hidden_dim)

    def forward(self, states):
        """(q1_list, q2_list) 각각 list[n_agents] of (B, action_n)"""
        return self.Q1(states), self.Q2(states)


# ─────────────────────────────────────────────────────────────
# 3. Memory  [출처: rltorch/memory/base.py]
# ─────────────────────────────────────────────────────────────

class Memory:
    """
    논문: Memory  (memory/base.py  전체)
    변경: actions shape (1,) → (n_agents,) 정수 배열

    논문과 동일한 인터페이스:
        append(state, action, reward, next_state, done)
        sample(batch_size) → (states, actions, rewards, next_states, dones)
        __len__()
    """

    def __init__(self, capacity, obs_dim, n_agents, device):
        self.capacity = int(capacity)
        self.device   = device

        # 논문 L12~13: 원형 버퍼 포인터
        self._n = 0
        self._p = 0

        # 논문 L65~69: numpy 버퍼 (actions만 shape 변경)
        self.states      = np.empty((self.capacity, obs_dim),  dtype=np.float32)
        self.actions     = np.empty((self.capacity, n_agents), dtype=np.int64)
        self.rewards     = np.empty((self.capacity, 1),        dtype=np.float32)
        self.next_states = np.empty((self.capacity, obs_dim),  dtype=np.float32)
        self.dones       = np.empty((self.capacity, 1),        dtype=np.float32)

    def append(self, state, action, reward, next_state, done):
        """논문 _append() L22~32 그대로"""
        self.states[self._p]      = state
        self.actions[self._p]     = action
        self.rewards[self._p]     = float(reward)
        self.next_states[self._p] = next_state
        self.dones[self._p]       = float(done)

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        """논문 _sample() L39~55 그대로 (이미지 처리 분기만 제거)"""
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states      = torch.FloatTensor(self.states[indices]).to(self.device)
        actions     = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards     = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones       = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n


# ─────────────────────────────────────────────────────────────
# 4. Agent  [출처: base.py + learner.py + actor.py 통합]
# ─────────────────────────────────────────────────────────────

class SacDiscreteAgent:
    """
    논문: SacDiscreteAgent (base.py) + SacDiscreteLearner (learner.py) 통합
    분산 처리(Actor/Learner 분리) → 단일 프로세스로 단순화

    유지한 메서드명:
        explore(), exploit()          ← base.py
        calc_current_q()              ← base.py
        calc_target_q()               ← base.py
        calc_critic_loss()            ← learner.py
        calc_policy_loss()            ← learner.py
        calc_entropy_loss()           ← learner.py
        learn()                       ← learner.py

    추가한 메서드 (MAAC 호환):
        prep_training(), prep_rollouts()
    """

    def __init__(self, obs_dim, action_n, n_agents,
                 gamma=0.99, tau=0.005, lr=0.0003, grad_clip=5.0,
                 pol_hidden_dim=256, critic_hidden_dim=256,
                 **kwargs):

        self.gamma     = gamma
        self.tau       = tau
        self.grad_clip = grad_clip
        self.n_agents  = n_agents
        self.action_n  = action_n
        self.niter     = 0

        # ── 네트워크 ─────────────────────────────────────────────
        self.policy        = CategoricalPolicy(obs_dim, action_n, n_agents, pol_hidden_dim)
        self.critic        = TwinedDiscreteQNetwork(obs_dim, action_n, n_agents, critic_hidden_dim)
        self.critic_target = TwinedDiscreteQNetwork(obs_dim, action_n, n_agents, critic_hidden_dim)
        hard_update(self.critic_target, self.critic)
        self.critic_target.eval()

        # ── 옵티마이저  [learner.py L48~51: Q1, Q2 각각 독립] ────
        self.policy_optim = Adam(self.policy.parameters(),    lr=lr)
        self.q1_optim     = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim     = Adam(self.critic.Q2.parameters(), lr=lr)

        # ── 학습 가능한 온도 α  [learner.py L52~56] ──────────────
        # 논문: target_entropy = log(n_actions) * 0.98
        # n_agents헤드 적용: n_agents * 0.98 * log(action_n)
        self.target_entropy = n_agents * 0.98 * np.log(action_n)
        self.log_alpha      = torch.zeros(1, requires_grad=True)
        self.alpha          = self.log_alpha.exp().detach()
        self.alpha_optim    = Adam([self.log_alpha], lr=lr)

        # device 추적
        self.pol_dev    = 'cpu'
        self.critic_dev = 'cpu'

    # ── 행동 선택  [base.py L29~41] ─────────────────────────────

    def explore(self, state):
        """논문 base.py explore() — 확률적 행동 (학습 중 탐험용)"""
        with torch.no_grad():
            actions, _, _, _ = self.policy.sample(state)
        return actions[0].cpu().numpy()   # (n_agents,) 정수

    def exploit(self, state):
        """논문 base.py exploit() — 탐욕적 행동 (평가용)"""
        with torch.no_grad():
            _, _, _, greedy = self.policy.sample(state)
        return greedy[0].cpu().numpy()    # (n_agents,) 정수

    # ── Q 계산  [base.py L43~62] ────────────────────────────────

    def calc_current_q(self, states, actions):
        """
        논문 base.py L43~47 calc_current_q() — 실제 취한 행동의 Q값 추출

        논문:
            curr_q1, curr_q2 = self.critic(states)
            curr_q1 = curr_q1.gather(1, actions.long())
            curr_q2 = curr_q2.gather(1, actions.long())

        n_agents헤드 확장:
            충전기 i : Q_i(s).gather(1, a_i) → (B,1)
            합산     : Σ_i Q_i(s)[a_i]       → (B,1)
        """
        q1_list, q2_list = self.critic(states)

        curr_q1 = sum(
            q1_list[i].gather(1, actions[:, i:i+1])
            for i in range(self.n_agents)
        )
        curr_q2 = sum(
            q2_list[i].gather(1, actions[:, i:i+1])
            for i in range(self.n_agents)
        )
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        """
        논문 base.py L50~62 calc_target_q() — 이산 SAC 핵심 수식

        논문:
            next_q = torch.min(next_q1, next_q2)
            next_q = next_action_probs * (next_q - alpha * log_next_action_probs)
            next_q = next_q.mean(dim=1).unsqueeze(-1)   ← 논문 코드 (mean 버그)
            target = rewards + (1-dones) * gamma * next_q

        수정: .mean() → .sum()  (수학적으로 올바른 기댓값 E_a[...])

        n_agents헤드 확장:
            next_v_i = Σ_a π_i(a|s') * [min(Q1_i, Q2_i)(s',a) - α*logπ_i(a|s')]
            next_v   = Σ_i next_v_i
        """
        with torch.no_grad():
            _, next_probs, next_log_probs, _ = self.policy.sample(next_states)
            next_q1_list, next_q2_list = self.critic_target(next_states)

            alpha = self.alpha.to(next_states.device)

            next_v = 0.0
            for i in range(self.n_agents):
                next_q_i = torch.min(next_q1_list[i], next_q2_list[i])  # (B, action_n)
                next_v  += (
                    next_probs[i] * (next_q_i - alpha * next_log_probs[i])
                ).sum(dim=1, keepdim=True)                               # (B, 1)

            target_q = rewards + (1.0 - dones) * self.gamma * next_v    # (B, 1)

        return target_q

    # ── 손실 계산  [learner.py L151~179] ────────────────────────

    def calc_critic_loss(self, batch):
        """논문 learner.py L151~161 calc_critic_loss()"""
        states, actions, rewards, next_states, dones = batch

        curr_q1, curr_q2 = self.calc_current_q(states, actions)
        target_q         = self.calc_target_q(states, actions, rewards, next_states, dones)

        q1_loss = (curr_q1 - target_q).pow(2).mean()
        q2_loss = (curr_q2 - target_q).pow(2).mean()
        return q1_loss, q2_loss

    def calc_policy_loss(self, batch):
        """
        논문 learner.py L163~173 calc_policy_loss()

        논문:
            q = alpha * log_action_probs - torch.min(q1, q2)
            inside_term = torch.sum(action_probs * q, dim=1, keepdim=True)
            policy_loss = inside_term.mean()
            entropies = -torch.sum(action_probs * log_action_probs, dim=1)
        """
        states = batch[0]

        _, action_probs, log_action_probs, _ = self.policy.sample(states)

        with torch.no_grad():
            q1_list, q2_list = self.critic(states)

        alpha = self.alpha.to(states.device)

        policy_loss = 0.0
        entropies   = 0.0

        for i in range(self.n_agents):
            min_q_i = torch.min(q1_list[i], q2_list[i])   # (B, action_n)

            # 논문 L168~170
            q_i          = alpha * log_action_probs[i] - min_q_i
            inside_term  = (action_probs[i] * q_i).sum(dim=1, keepdim=True)
            policy_loss += inside_term.mean()

            # 논문 L172~173
            entropies -= (action_probs[i] * log_action_probs[i]).sum(dim=1, keepdim=True)

        return policy_loss, entropies

    def calc_entropy_loss(self, entropies):
        """논문 learner.py L175~179 calc_entropy_loss()"""
        entropy_loss = -(
            self.log_alpha.to(entropies.device)
            * (self.target_entropy - entropies).detach()
        ).mean()
        return entropy_loss

    # ── 학습  [learner.py L105~131 learn()] ─────────────────────

    def learn(self, batch, logger=None):
        """
        논문 learner.py L105~131 learn() 구조 그대로.

        논문 순서:
            1. calc_critic_loss  → update Q1, Q2
            2. calc_policy_loss  → update policy
            3. calc_entropy_loss → update alpha
            4. soft_update critic_target
        """
        # 1. Critic 업데이트  [learner.py L113~114, L118~123]
        q1_loss, q2_loss = self.calc_critic_loss(batch)
        update_params(self.q1_optim, q1_loss, self.grad_clip,
                      self.critic.Q1, retain_graph=True)
        update_params(self.q2_optim, q2_loss, self.grad_clip,
                      self.critic.Q2)

        # 2. Policy 업데이트  [learner.py L115, L119]
        policy_loss, entropies = self.calc_policy_loss(batch)
        update_params(self.policy_optim, policy_loss, self.grad_clip,
                      self.policy)

        # 3. Alpha 업데이트  [learner.py L116, L120]
        entropy_loss = self.calc_entropy_loss(entropies)
        update_params(self.alpha_optim, entropy_loss, grad_clip=None)
        self.alpha = self.log_alpha.exp().detach().cpu()

        if logger is not None:
            logger.add_scalar('losses/q1',      q1_loss.item(),               self.niter)
            logger.add_scalar('losses/q2',      q2_loss.item(),               self.niter)
            logger.add_scalar('losses/policy',  policy_loss.item(),           self.niter)
            logger.add_scalar('losses/alpha',   entropy_loss.item(),          self.niter)
            logger.add_scalar('stats/alpha',    self.alpha.item(),            self.niter)
            logger.add_scalar('stats/entropy',  entropies.detach().mean().item(), self.niter)

        self.niter += 1

    # ── 타깃 업데이트  [learner.py L191: soft_update] ───────────

    def update_all_targets(self):
        """논문 learner.py interval() → soft_update 호출"""
        soft_update(self.critic_target, self.critic, self.tau)

    # ── 디바이스 관리  (MAAC attention_sac.py 패턴) ──────────────

    def prep_training(self, device='gpu'):
        fn = (lambda x: x.cuda()) if device == 'gpu' else (lambda x: x.cpu())
        self.policy.train()
        self.critic.train()
        self.critic_target.train()
        if self.pol_dev != device:
            self.policy    = fn(self.policy)
            self.pol_dev   = device
        if self.critic_dev != device:
            self.critic        = fn(self.critic)
            self.critic_target = fn(self.critic_target)
            self.critic_dev    = device

    def prep_rollouts(self, device='cpu'):
        fn = (lambda x: x.cuda()) if device == 'gpu' else (lambda x: x.cpu())
        self.policy.eval()
        if self.pol_dev != device:
            self.policy  = fn(self.policy)
            self.pol_dev = device

    # ── 저장 / 로드 ──────────────────────────────────────────────

    def save(self, filename):
        self.prep_training(device='cpu')
        torch.save({
            'init_dict':     self.init_dict,
            'policy':        self.policy.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'policy_optim':  self.policy_optim.state_dict(),
            'q1_optim':      self.q1_optim.state_dict(),
            'q2_optim':      self.q2_optim.state_dict(),
            'log_alpha':     self.log_alpha.detach(),
        }, filename)

    @classmethod
    def init_from_env(cls, wrapped_env, **kwargs):
        obs_dim  = wrapped_env.observation_space.shape[0]
        action_n = int(wrapped_env.action_space.nvec[0])
        n_agents = int(len(wrapped_env.action_space.nvec))
        instance = cls(obs_dim, action_n, n_agents, **kwargs)
        instance.init_dict = dict(
            obs_dim=obs_dim, action_n=action_n, n_agents=n_agents, **kwargs)
        return instance

    @classmethod
    def init_from_save(cls, filename):
        d        = torch.load(filename)
        instance = cls(**d['init_dict'])
        instance.init_dict = d['init_dict']
        instance.policy.load_state_dict(d['policy'])
        instance.critic.load_state_dict(d['critic'])
        instance.critic_target.load_state_dict(d['critic_target'])
        instance.policy_optim.load_state_dict(d['policy_optim'])
        instance.q1_optim.load_state_dict(d['q1_optim'])
        instance.q2_optim.load_state_dict(d['q2_optim'])
        instance.log_alpha.data = d['log_alpha']
        instance.alpha = instance.log_alpha.exp().detach()
        return instance


# ─────────────────────────────────────────────────────────────
# 5. 학습 루프  [출처: actor.py act_episode() + main.py]
# ─────────────────────────────────────────────────────────────

def run(config):
    """
    논문 actor.py act_episode() 구조를 단일 프로세스로 단순화.
    디렉토리 관리·로깅은 main.py 구조 그대로.
    """
    # 디렉토리 관리 (main.py 그대로)
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst    = [int(str(f.name).split('run')[1])
                   for f in model_dir.iterdir() if str(f.name).startswith('run')]
        run_num = max(exst) + 1 if exst else 1

    run_dir = model_dir / ('run%i' % run_num)
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    seed = config.seed if config.seed is not None else run_num
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 환경 (MA 환경 → Single-Agent 래핑)
    ma_env = make_env(config.env_id, discrete_action=True,
                      arrival_mode=config.arrival_mode)
    ma_env.seed(seed)
    env = MultiToSingleWrapper(ma_env)

    device = torch.device(
        'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')

    model = SacDiscreteAgent.init_from_env(
        env,
        gamma=config.gamma, tau=config.tau, lr=config.lr,
        grad_clip=config.grad_clip,
        pol_hidden_dim=config.pol_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim)

    memory = Memory(
        config.buffer_length,
        obs_dim=env.observation_space.shape[0],
        n_agents=env.n_agents,
        device=device)

    import time as _time
    t          = 0
    best_rew   = -float('inf')
    t_start    = _time.time()

    for ep_i in range(config.n_episodes):

        # 논문 actor.py act_episode() L88~89
        state  = env.reset()
        ep_rew = 0.0
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):

            # 논문 actor.py L23~26: start_steps 동안 랜덤 탐험
            if t < config.start_steps:
                action = env.action_space.sample()
            else:
                obs_t  = torch.FloatTensor(state).unsqueeze(0)
                action = model.explore(obs_t)          # (n_agents,) 정수

            # 논문 actor.py L93~94
            next_state, reward, done, _ = env.step(action)
            ep_rew += reward

            # 논문 actor.py L115~121: 메모리에 저장
            memory.append(state, action, reward, next_state, done)
            state = next_state
            t    += 1

            if done:
                state = env.reset()

            # 업데이트 (main.py 타이밍 그대로)
            if len(memory) >= config.batch_size and t % config.steps_per_update == 0:
                model.prep_training(device='gpu' if config.use_gpu else 'cpu')
                for _ in range(config.num_updates):
                    batch = memory.sample(config.batch_size)
                    model.learn(batch, logger=logger)          # learner.py learn()
                    model.update_all_targets()                 # learner.py interval()
                model.prep_rollouts(device='cpu')

        logger.add_scalar('all/mean_episode_reward',  ep_rew, ep_i)
        logger.add_scalar('all/mean_episode_rewards', ep_rew, ep_i)  # visualize.py 호환

        if ep_rew > best_rew:
            best_rew = ep_rew
            model.prep_rollouts(device='cpu')
            model.save(run_dir / 'model_best.pt')

        if ep_i % config.print_interval == 0 or ep_i == config.n_episodes - 1:
            elapsed  = _time.time() - t_start
            eps_done = ep_i + 1
            eps_left = config.n_episodes - eps_done
            eta      = elapsed / eps_done * eps_left if eps_done > 0 else 0

            def _fmt(s):
                s = int(s)
                h, r = divmod(s, 3600)
                m, s = divmod(r, 60)
                return (f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s")

            print(f"  Episode {eps_done:5d}/{config.n_episodes} | "
                  f"Reward: {ep_rew:10.1f} | "
                  f"Best: {best_rew:10.1f} | "
                  f"Time: {_fmt(elapsed):>8s} | "
                  f"ETA: {_fmt(eta)}")

        if ep_i % config.save_interval == 0:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


# ─────────────────────────────────────────────────────────────
# 실행 인자
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discrete SAC — EV Charging (Christodoulou 2019 기반)')
    parser.add_argument('--env_id',        default='ev_charging', type=str)
    parser.add_argument('--model_name',    default='sac_model',   type=str)
    parser.add_argument('--buffer_length',     default=int(1e6), type=int)
    parser.add_argument('--n_episodes',        default=30000,    type=int)
    parser.add_argument('--episode_length',    default=144,      type=int)
    parser.add_argument('--start_steps',       default=10000,    type=int)
    parser.add_argument('--steps_per_update',  default=144,      type=int)
    parser.add_argument('--num_updates',       default=4,        type=int)
    parser.add_argument('--batch_size',        default=1024,     type=int)
    parser.add_argument('--save_interval',     default=1000,     type=int)
    parser.add_argument('--print_interval',    default=50,       type=int)
    parser.add_argument('--pol_hidden_dim',    default=256,      type=int)
    parser.add_argument('--critic_hidden_dim', default=256,      type=int)
    parser.add_argument('--lr',                default=0.0003,   type=float)
    parser.add_argument('--tau',               default=0.005,    type=float)
    parser.add_argument('--gamma',             default=0.99,     type=float)
    parser.add_argument('--grad_clip',         default=5.0,      type=float)
    parser.add_argument('--seed',              default=None,     type=int,
                        help='Random seed (기본: run_num 자동 사용)')
    parser.add_argument('--use_gpu',           action='store_true')
    parser.add_argument('--arrival_mode',      default='normal_10', type=str,
                        choices=['normal_10', 'extreme', 'smooth', 'low'])

    config = parser.parse_args()
    run(config)
