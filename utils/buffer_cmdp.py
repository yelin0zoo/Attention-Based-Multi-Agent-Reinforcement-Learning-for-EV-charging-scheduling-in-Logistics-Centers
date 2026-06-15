"""CMDP 리플레이 버퍼 (제약 비용 추가 저장)."""

import numpy as np
from torch import Tensor
from torch.autograd import Variable


class CMDPReplayBuffer:
    """저장: (obs, action, reward_corrected, next_obs, done, costs[K])."""

    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, n_constraints=2):
        self.max_steps   = max_steps
        self.num_agents  = num_agents
        self.n_constraints = n_constraints

        self.obs_buffs      = [np.zeros((max_steps, d), dtype=np.float32) for d in obs_dims]
        self.ac_buffs       = [np.zeros((max_steps, d), dtype=np.float32) for d in ac_dims]
        self.rew_buffs      = [np.zeros(max_steps,      dtype=np.float32) for _ in range(num_agents)]
        self.next_obs_buffs = [np.zeros((max_steps, d), dtype=np.float32) for d in obs_dims]
        self.done_buffs     = [np.zeros(max_steps,      dtype=np.uint8)   for _ in range(num_agents)]

        # cost_buffs[k][i]: constraint k, agent i
        self.cost_buffs = [
            [np.zeros(max_steps, dtype=np.float32) for _ in range(num_agents)]
            for _ in range(n_constraints)
        ]

        self.filled_i = 0
        self.curr_i   = 0

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones, costs):
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

        for i in range(self.num_agents):
            self.obs_buffs[i][self.curr_i:self.curr_i + nentries]      = np.vstack(observations[:, i])
            self.ac_buffs[i][self.curr_i:self.curr_i + nentries]       = actions[i]
            self.rew_buffs[i][self.curr_i:self.curr_i + nentries]      = rewards[:, i]
            self.next_obs_buffs[i][self.curr_i:self.curr_i + nentries] = np.vstack(next_observations[:, i])
            self.done_buffs[i][self.curr_i:self.curr_i + nentries]     = dones[:, i]

        for k in range(self.n_constraints):
            for i in range(self.num_agents):
                self.cost_buffs[k][i][self.curr_i:self.curr_i + nentries] = costs[k][:, i]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        """반환: (obs, acs, rews, next_obs, dones, costs[K][N_agents])."""
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

        # 제약 비용은 정규화 없이 원본 값 그대로
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
