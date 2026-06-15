import numpy as np
import gym


class MultiToSingleWrapper(gym.Env):
    """
    Adapts a multi-agent environment (list-of-N obs/rew/done) into a
    standard single-agent Gym interface without modifying the underlying env.

    Observations : all N agent obs concatenated   → 1-D flat vector
    Actions      : MultiDiscrete([n]*N) integer    → split & one-hot into MA env
    Reward       : sum of per-agent rewards        → scalar
    Done         : any agent done                  → bool
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, ma_env):
        super().__init__()
        self.ma_env   = ma_env
        self.n_agents = len(ma_env.action_space)
        self.action_n  = ma_env.action_space[0].n    # acsp.n  (discrete levels)

        # flat observation space (concatenated)
        total_obs_dim = sum(sp.shape[0] for sp in ma_env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_obs_dim,), dtype=np.float32)

        # MultiDiscrete: 14 independent choices each in [0, action_n)
        self.action_space = gym.spaces.MultiDiscrete(
            [self.action_n] * self.n_agents)

        self._obs_dims = [sp.shape[0] for sp in ma_env.observation_space]

    # ── Gym API ────────────────────────────────────────────────────────
    def seed(self, seed=None):
        return self.ma_env.seed(seed)

    def reset(self):
        """Returns concatenated flat obs vector."""
        obs_list = self.ma_env.reset()                  # list[N] of np arrays
        return np.concatenate(obs_list).astype(np.float32)

    def step(self, joint_action):
        """
        Args:
            joint_action : (N,) integer array, each value in [0, action_n)
        Returns:
            next_obs (np.ndarray) : flat concatenated observation
            reward   (float)      : sum of per-agent rewards
            done     (bool)       : any agent is done
            info     (dict)
        """
        action_list = self._to_onehot_list(joint_action)
        next_obs_list, rew_list, done_list, info = self.ma_env.step(action_list)

        next_obs = np.concatenate(next_obs_list).astype(np.float32)
        reward   = float(sum(rew_list))
        done     = bool(any(done_list))
        return next_obs, reward, done, info

    def render(self, mode='human'):
        return self.ma_env.render(mode)

    def close(self):
        if hasattr(self.ma_env, 'close'):
            return self.ma_env.close()

    # ── helpers ────────────────────────────────────────────────────────
    def _to_onehot_list(self, joint_action):
        """Convert integer joint action → list of one-hot np arrays for MA env."""
        action_list = []
        for a in joint_action:
            one_hot = np.zeros(self.action_n, dtype=np.float32)
            one_hot[int(a)] = 1.0
            action_list.append(one_hot)
        return action_list

    def split_obs(self, flat_obs):
        """Utility: recover per-agent obs list from a flat concatenated obs."""
        parts, idx = [], 0
        for d in self._obs_dims:
            parts.append(flat_obs[idx: idx + d])
            idx += d
        return parts
