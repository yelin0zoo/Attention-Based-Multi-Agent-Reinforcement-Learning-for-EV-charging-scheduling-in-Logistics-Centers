"""
CMDP 리플레이 버퍼

기존 ReplayBuffer를 확장하여 제약 비용(constraint cost)을 함께 저장.
  - C1: 과부하 비용 (overload)
  - C2: 미충전 비용 (undercharge)

기존 buffer.py는 수정하지 않음.
"""

import numpy as np
from torch import Tensor
from torch.autograd import Variable


class CMDPReplayBuffer:
    """
    CMDP 멀티에이전트 리플레이 버퍼

    저장 항목:
        (obs, action, reward_corrected, next_obs, done, [cost_c1, cost_c2])

    reward_corrected: 과부하/미충전 페널티를 제거한 순수 보상
    cost_c1, cost_c2: 제약 비용 (에이전트별)
    """

    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, n_constraints=2):
        self.max_steps   = max_steps
        self.num_agents  = num_agents
        self.n_constraints = n_constraints

        # ── 기본 버퍼 ─────────────────────────────────────────────
        self.obs_buffs      = [np.zeros((max_steps, d), dtype=np.float32) for d in obs_dims]
        self.ac_buffs       = [np.zeros((max_steps, d), dtype=np.float32) for d in ac_dims]
        self.rew_buffs      = [np.zeros(max_steps,      dtype=np.float32) for _ in range(num_agents)]
        self.next_obs_buffs = [np.zeros((max_steps, d), dtype=np.float32) for d in obs_dims]
        self.done_buffs     = [np.zeros(max_steps,      dtype=np.uint8)   for _ in range(num_agents)]

        # ── 제약 비용 버퍼: cost_buffs[k][i] → constraint k, agent i ──
        self.cost_buffs = [
            [np.zeros(max_steps, dtype=np.float32) for _ in range(num_agents)]
            for _ in range(n_constraints)
        ]

        self.filled_i = 0
        self.curr_i   = 0

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones, costs):
        """
        경험 데이터 저장

        인자:
            observations     : [n_envs, n_agents, obs_dim]
            actions          : list of n_agents [n_envs, ac_dim]
            rewards          : [n_envs, n_agents]  (제약 비용 제거된 보상)
            next_observations: [n_envs, n_agents, obs_dim]
            dones            : [n_envs, n_agents]
            costs            : list of n_constraints [n_envs, n_agents]
        """
        nentries = observations.shape[0]

        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i
            for i in range(self.num_agents):
                self.obs_buffs[i]      = np.roll(self.obs_buffs[i],      rollover, axis=0)
                self.ac_buffs[i]       = np.roll(self.ac_buffs[i],       rollover, axis=0)
                self.rew_buffs[i]      = np.roll(self.rew_buffs[i],      rollover)
                self.next_obs_buffs[i] = np.roll(self.next_obs_buffs[i], rollover, axis=0)
                self.done_buffs[i]     = np.roll(self.done_buffs[i],     rollover)
            for k in range(self.n_constraints):
                for i in range(self.num_agents):
                    self.cost_buffs[k][i] = np.roll(self.cost_buffs[k][i], rollover)
            self.curr_i   = 0
            self.filled_i = self.max_steps

        # 기본 데이터 저장
        for i in range(self.num_agents):
            self.obs_buffs[i][self.curr_i:self.curr_i + nentries]      = np.vstack(observations[:, i])
            self.ac_buffs[i][self.curr_i:self.curr_i + nentries]       = actions[i]
            self.rew_buffs[i][self.curr_i:self.curr_i + nentries]      = rewards[:, i]
            self.next_obs_buffs[i][self.curr_i:self.curr_i + nentries] = np.vstack(next_observations[:, i])
            self.done_buffs[i][self.curr_i:self.curr_i + nentries]     = dones[:, i]

        # 제약 비용 저장
        for k in range(self.n_constraints):
            for i in range(self.num_agents):
                self.cost_buffs[k][i][self.curr_i:self.curr_i + nentries] = costs[k][:, i]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        """
        미니배치 샘플링

        반환:
            (obs, acs, rews, next_obs, dones, costs)
            costs: list of n_constraints × list of n_agents Tensors [N]
        """
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)

        cast = lambda x: (Variable(Tensor(x), requires_grad=False).cuda()
                          if to_gpu
                          else Variable(Tensor(x), requires_grad=False))

        if norm_rews:
            ret_rews = [
                cast((self.rew_buffs[i][inds] - self.rew_buffs[i][:self.filled_i].mean()) /
                     (self.rew_buffs[i][:self.filled_i].std() + 1e-6))
                for i in range(self.num_agents)
            ]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]

        # 제약 비용: 정규화 없이 원본 값 사용
        ret_costs = [
            [cast(self.cost_buffs[k][i][inds]) for i in range(self.num_agents)]
            for k in range(self.n_constraints)
        ]

        return (
            [cast(self.obs_buffs[i][inds])      for i in range(self.num_agents)],
            [cast(self.ac_buffs[i][inds])        for i in range(self.num_agents)],
            ret_rews,
            [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
            [cast(self.done_buffs[i][inds])     for i in range(self.num_agents)],
            ret_costs,
        )
