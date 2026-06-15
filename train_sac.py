"""
Single-Agent Discrete SAC (EV 충전 베이스라인)

Christodoulou (2019), "Soft Actor-Critic for Discrete Action Settings"의
rltorch 레퍼런스 구현을 단일 프로세스 + n_agents 헤드 구조로 적응.
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


def update_params(optim, loss, grad_clip, network=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None and network is not None:
        nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optim.step()


class CategoricalPolicy(nn.Module):
    """공유 MLP + n_agents개 독립 카테고리컬 헤드."""

    def __init__(self, obs_dim, action_n, n_agents, hidden_dim=256):
        super().__init__()
        self.n_agents = n_agents
        self.action_n = action_n

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_n) for _ in range(n_agents)
        ])

    def forward(self, states):
        h = self.net(states)
        return [F.softmax(head(h), dim=1) for head in self.heads]

    def sample(self, states):
        """반환: (actions[B,n_agents], probs_list, log_probs_list, greedy[B,n_agents])."""
        h = self.net(states)

        actions_list, probs_list, log_probs_list, greedy_list = [], [], [], []

        for head in self.heads:
            probs = F.softmax(head(h), dim=1)
            log_probs = torch.log(probs + (probs == 0.0).float() * 1e-8)

            categorical     = Categorical(probs)
            actions         = categorical.sample().view(-1, 1)
            greedy_actions  = torch.argmax(probs, dim=1, keepdim=True)

            actions_list.append(actions)
            probs_list.append(probs)
            log_probs_list.append(log_probs)
            greedy_list.append(greedy_actions)

        return (
            torch.cat(actions_list, dim=1),
            probs_list,
            log_probs_list,
            torch.cat(greedy_list,  dim=1),
        )


class DiscreteQNetwork(nn.Module):
    """공유 MLP + n_agents개 독립 Q 헤드 (Q(s)[a] 동시 출력)."""

    def __init__(self, obs_dim, action_n, n_agents, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_n) for _ in range(n_agents)
        ])

    def forward(self, states):
        h = self.net(states)
        return [head(h) for head in self.heads]


class TwinedDiscreteQNetwork(nn.Module):
    """Q1, Q2 듀얼 크리틱 (각자 독립 옵티마이저 적용)."""

    def __init__(self, obs_dim, action_n, n_agents, hidden_dim=256):
        super().__init__()
        self.Q1 = DiscreteQNetwork(obs_dim, action_n, n_agents, hidden_dim)
        self.Q2 = DiscreteQNetwork(obs_dim, action_n, n_agents, hidden_dim)

    def forward(self, states):
        return self.Q1(states), self.Q2(states)


class Memory:
    def __init__(self, capacity, obs_dim, n_agents, device):
        self.capacity = int(capacity)
        self.device   = device

        self._n = 0
        self._p = 0

        self.states      = np.empty((self.capacity, obs_dim),  dtype=np.float32)
        self.actions     = np.empty((self.capacity, n_agents), dtype=np.int64)
        self.rewards     = np.empty((self.capacity, 1),        dtype=np.float32)
        self.next_states = np.empty((self.capacity, obs_dim),  dtype=np.float32)
        self.dones       = np.empty((self.capacity, 1),        dtype=np.float32)

    def append(self, state, action, reward, next_state, done):
        self.states[self._p]      = state
        self.actions[self._p]     = action
        self.rewards[self._p]     = float(reward)
        self.next_states[self._p] = next_state
        self.dones[self._p]       = float(done)

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states      = torch.FloatTensor(self.states[indices]).to(self.device)
        actions     = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards     = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones       = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n


class SacDiscreteAgent:
    """이산 SAC (Christodoulou 2019) 단일 프로세스 구현 + n_agents 헤드."""

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

        self.policy        = CategoricalPolicy(obs_dim, action_n, n_agents, pol_hidden_dim)
        self.critic        = TwinedDiscreteQNetwork(obs_dim, action_n, n_agents, critic_hidden_dim)
        self.critic_target = TwinedDiscreteQNetwork(obs_dim, action_n, n_agents, critic_hidden_dim)
        hard_update(self.critic_target, self.critic)
        self.critic_target.eval()

        self.policy_optim = Adam(self.policy.parameters(),    lr=lr)
        self.q1_optim     = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim     = Adam(self.critic.Q2.parameters(), lr=lr)

        # target_entropy = n_agents · 0.98 · log(action_n)  (per-head 합산)
        self.target_entropy = n_agents * 0.98 * np.log(action_n)
        self.log_alpha      = torch.zeros(1, requires_grad=True)
        self.alpha          = self.log_alpha.exp().detach()
        self.alpha_optim    = Adam([self.log_alpha], lr=lr)

        self.pol_dev    = 'cpu'
        self.critic_dev = 'cpu'

    def explore(self, state):
        with torch.no_grad():
            actions, _, _, _ = self.policy.sample(state)
        return actions[0].cpu().numpy()

    def exploit(self, state):
        with torch.no_grad():
            _, _, _, greedy = self.policy.sample(state)
        return greedy[0].cpu().numpy()

    def calc_current_q(self, states, actions):
        """Σ_i Q_i(s)[a_i] (헤드별 합산)"""
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
        """타겟: r + γ·(1−d)·Σ_i E_a[min(Q1_i, Q2_i) − α·log π_i(a|s')]"""
        with torch.no_grad():
            _, next_probs, next_log_probs, _ = self.policy.sample(next_states)
            next_q1_list, next_q2_list = self.critic_target(next_states)

            alpha = self.alpha.to(next_states.device)

            next_v = 0.0
            for i in range(self.n_agents):
                next_q_i = torch.min(next_q1_list[i], next_q2_list[i])
                next_v  += (
                    next_probs[i] * (next_q_i - alpha * next_log_probs[i])
                ).sum(dim=1, keepdim=True)

            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        return target_q

    def calc_critic_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        curr_q1, curr_q2 = self.calc_current_q(states, actions)
        target_q         = self.calc_target_q(states, actions, rewards, next_states, dones)

        q1_loss = (curr_q1 - target_q).pow(2).mean()
        q2_loss = (curr_q2 - target_q).pow(2).mean()
        return q1_loss, q2_loss

    def calc_policy_loss(self, batch):
        states = batch[0]

        _, action_probs, log_action_probs, _ = self.policy.sample(states)

        with torch.no_grad():
            q1_list, q2_list = self.critic(states)

        alpha = self.alpha.to(states.device)

        policy_loss = 0.0
        entropies   = 0.0

        for i in range(self.n_agents):
            min_q_i = torch.min(q1_list[i], q2_list[i])

            q_i          = alpha * log_action_probs[i] - min_q_i
            inside_term  = (action_probs[i] * q_i).sum(dim=1, keepdim=True)
            policy_loss += inside_term.mean()

            entropies -= (action_probs[i] * log_action_probs[i]).sum(dim=1, keepdim=True)

        return policy_loss, entropies

    def calc_entropy_loss(self, entropies):
        entropy_loss = -(
            self.log_alpha.to(entropies.device)
            * (self.target_entropy - entropies).detach()
        ).mean()
        return entropy_loss

    def learn(self, batch, logger=None):
        q1_loss, q2_loss = self.calc_critic_loss(batch)
        update_params(self.q1_optim, q1_loss, self.grad_clip,
                      self.critic.Q1, retain_graph=True)
        update_params(self.q2_optim, q2_loss, self.grad_clip,
                      self.critic.Q2)

        policy_loss, entropies = self.calc_policy_loss(batch)
        update_params(self.policy_optim, policy_loss, self.grad_clip,
                      self.policy)

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

    def update_all_targets(self):
        soft_update(self.critic_target, self.critic, self.tau)

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


def run(config):
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

    # 멀티에이전트 환경 → 단일 에이전트 인터페이스로 래핑
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
        state  = env.reset()
        ep_rew = 0.0
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            if t < config.start_steps:
                action = env.action_space.sample()
            else:
                obs_t  = torch.FloatTensor(state).unsqueeze(0)
                action = model.explore(obs_t)

            next_state, reward, done, _ = env.step(action)
            ep_rew += reward

            memory.append(state, action, reward, next_state, done)
            state = next_state
            t    += 1

            if done:
                state = env.reset()

            if len(memory) >= config.batch_size and t % config.steps_per_update == 0:
                model.prep_training(device='gpu' if config.use_gpu else 'cpu')
                for _ in range(config.num_updates):
                    batch = memory.sample(config.batch_size)
                    model.learn(batch, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')

        logger.add_scalar('all/mean_episode_reward',  ep_rew, ep_i)
        logger.add_scalar('all/mean_episode_rewards', ep_rew, ep_i)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discrete SAC — EV Charging')
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
