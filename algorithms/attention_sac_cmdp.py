"""
AttentionSAC_CMDP: MAAC의 CMDP (Constrained MDP) 확장 (Lagrangian Relaxation)

  - 제약 C1 (과부하): 비용 ≤ d1
  - 제약 C2 (미충전): 비용 ≤ d2

정책 손실: L(π) = E[log π · (log π / α − A_r + Σ_k λ_k · A_c_k)]
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam

from algorithms.attention_sac import AttentionSAC
from utils.critics import AttentionCritic
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients

MSELoss = torch.nn.MSELoss()


class AttentionSACCMDP(AttentionSAC):
    """AttentionSAC + 제약 크리틱(Q_c) + 라그랑지안 승수(λ)."""

    def __init__(self,
                 agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 n_constraints=2,
                 constraint_thresholds=None,
                 lambda_lr=1e-3,
                 lambda_max=10.0,
                 **kwargs):
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

        if constraint_thresholds is None:
            self.constraint_thresholds = [0.0] * n_constraints
        else:
            self.constraint_thresholds = list(constraint_thresholds)

        self.lambda_lr  = lambda_lr
        self.lambda_max = lambda_max

        # λ는 closed-form 업데이트로 직접 수정 (requires_grad 불필요)
        self.lambdas = [torch.tensor(0.0, dtype=torch.float32)
                        for _ in range(n_constraints)]

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
        for cc, tcc in zip(self.constraint_critics,
                           self.target_constraint_critics):
            hard_update(tcc, cc)

        self.constraint_optimizers = [
            Adam(cc.parameters(), lr=q_lr, weight_decay=1e-3)
            for cc in self.constraint_critics
        ]

        self.constraint_critic_dev = 'cpu'
        self.trgt_constraint_critic_dev = 'cpu'

    def update_constraint_critics(self, sample, logger=None):
        """제약 크리틱 Q_c 업데이트 (엔트로피 항 없음)."""
        obs, acs, rews, next_obs, dones, costs = sample

        with torch.no_grad():
            next_acs = [pi(ob, return_log_pi=False)
                        for pi, ob in zip(self.target_policies, next_obs)]
            trgt_cc_in = list(zip(next_obs, next_acs))

        cc_losses = []
        for k, (cc, tcc, cc_opt) in enumerate(zip(
                self.constraint_critics,
                self.target_constraint_critics,
                self.constraint_optimizers)):

            cc_in = list(zip(obs, acs))
            cc_rets = cc(cc_in, regularize=True, logger=logger, niter=self.niter)

            with torch.no_grad():
                next_qcs = tcc(trgt_cc_in)

            qc_loss = 0.0
            for a_i, (pqc, regs), nqc in zip(
                    range(self.nagents), cc_rets, next_qcs):
                cost_ai = costs[k][a_i].view(-1, 1)
                done_ai = dones[a_i].view(-1, 1)

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

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
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

        critic_in   = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)

        cc_rets_all = []
        for cc in self.constraint_critics:
            cc_rets_all.append(cc(critic_in, return_all_q=True))

        pol_losses = []
        for a_i, probs, log_pi, pol_regs, (q_r, all_q_r) in zip(
                range(self.nagents), all_probs, all_log_pis,
                all_pol_regs, critic_rets):

            curr_agent = self.agents[a_i]

            v_r     = (all_q_r * probs).sum(dim=1, keepdim=True)
            pol_target_r = q_r - v_r

            # 제약 어드밴티지 합산: Σ_k λ_k · A_c_k
            constraint_penalty = torch.zeros_like(pol_target_r)
            for k, cc_rets in enumerate(cc_rets_all):
                q_c, all_q_c = cc_rets[a_i]
                v_c = (all_q_c * probs).sum(dim=1, keepdim=True)
                pol_target_c = q_c - v_c
                lam = self.lambdas[k].to(pol_target_c.device)
                constraint_penalty = constraint_penalty + lam * pol_target_c

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

    def update_lambdas(self, costs_batch):
        """λ_k ← clip(λ_k + lr_λ · (mean_cost_k − d_k), 0, λ_max)"""
        updated = []
        for k in range(self.n_constraints):
            mean_cost = torch.stack(
                [costs_batch[k][i].mean() for i in range(self.nagents)]
            ).mean().item()

            d_k   = self.constraint_thresholds[k]
            lam_k = self.lambdas[k].item()

            new_lam = lam_k + self.lambda_lr * (mean_cost - d_k)
            new_lam = max(0.0, min(self.lambda_max, new_lam))
            self.lambdas[k] = torch.tensor(new_lam, dtype=torch.float32)
            updated.append(new_lam)

        return updated

    def update_all_targets(self):
        super().update_all_targets()
        for cc, tcc in zip(self.constraint_critics,
                           self.target_constraint_critics):
            soft_update(tcc, cc, self.tau)

    def prep_training(self, device='gpu'):
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

        dev = torch.device('cuda' if device == 'gpu' else 'cpu')
        self.lambdas = [lam.to(dev) for lam in self.lambdas]

    def prep_rollouts(self, device='cpu'):
        super().prep_rollouts(device=device)
        dev = torch.device('cuda' if device == 'gpu' else 'cpu')
        self.lambdas = [lam.to(dev) for lam in self.lambdas]

    def save(self, filename):
        self.prep_training(device='cpu')
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
        save_dict = torch.load(filename, map_location='cpu')
        instance  = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']

        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            cp = save_dict['critic_params']
            instance.critic.load_state_dict(cp['critic'])
            instance.target_critic.load_state_dict(cp['target_critic'])
            instance.critic_optimizer.load_state_dict(cp['critic_optimizer'])

            if 'constraint_params' in save_dict:
                for k, ccp in enumerate(save_dict['constraint_params']):
                    instance.constraint_critics[k].load_state_dict(ccp['critic'])
                    instance.target_constraint_critics[k].load_state_dict(
                        ccp['target_critic'])
                    instance.constraint_optimizers[k].load_state_dict(
                        ccp['optimizer'])

        if 'lambdas' in save_dict:
            instance.lambdas = [
                torch.tensor(float(v), dtype=torch.float32)
                for v in save_dict['lambdas']
            ]

        return instance
