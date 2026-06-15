"""
EV 충전 CMDP-MAAC 학습 스크립트 (연속 비용 버전)

train_cmdp.py와의 차이점:
  - C1 비용: 이진(0/1) → 연속 (초과량 비율)
    cost_c1[i] = (actual_power[i] / total_power) * (overload_kw / GRID_LIMIT_KW)
  - d1 단위: 스텝당 평균 초과 비율 (0.0 = 완전 금지)
  - Q_c가 dense한 신호로 학습 → λ가 실제로 정책에 영향 가능

사용법:
  python train_cmdp_cont.py --model_name cmdp_v5_s1 --seed 1 \
      --arrival_mode normal_10 --n_episodes 20000 \
      --d1 0.0 --lambda_lr 0.1
"""
import sys
import io
import os
import time
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def train(config):
    """CMDP-MAAC 학습 실행 (연속 비용)"""
    import torch
    import numpy as np
    from pathlib import Path
    from torch.autograd import Variable

    from tensorboardX import SummaryWriter
    from gym.spaces import Box

    from utils.make_env import make_env
    from utils.buffer_cmdp import CMDPReplayBuffer
    from algorithms.attention_sac_cmdp import AttentionSACCMDP
    from envs.ev_charging.ev_charging_env import GRID_LIMIT_KW

    # ================================================================
    # 1. 환경 생성
    # ================================================================
    print("=" * 65)
    print("  EV Charging CMDP-MAAC Training  [Continuous Cost]")
    print("=" * 65)
    print(f"\n  Episodes      : {config.n_episodes}")
    print(f"  Episode length: {config.episode_length} steps (= 24 hours)")
    print(f"  Reward scale  : {config.reward_scale}")
    print(f"  Batch size    : {config.batch_size}")
    print(f"  GPU           : {config.use_gpu}")
    print(f"  Arrival mode  : {config.arrival_mode}")
    print(f"  GRID_LIMIT_KW : {GRID_LIMIT_KW} kW")

    n_constraints = 1 if config.d2 is None else 2
    constraint_thresholds = [config.d1] if config.d2 is None else [config.d1, config.d2]

    print(f"  C1 threshold  : d1 = {config.d1}  (과부하 p4 비용 평균, 0.0 = 완전 금지)")
    if config.d2 is not None:
        print(f"  C2 threshold  : d2 = {config.d2}  (강제출차, 0.0 = 완전 금지)")
    else:
        print(f"  강제출차      : 보상에 포함 (제약 없음)")
    print(f"  Lambda LR     : {config.lambda_lr}")
    print(f"  Lambda max    : {config.lambda_max}")
    print()

    env = make_env('ev_charging', discrete_action=True,
                   arrival_mode=config.arrival_mode)
    env.seed(config.seed)

    n_agents = len(env.action_space)
    obs_dims = [sp.shape[0] for sp in env.observation_space]
    ac_dims  = [sp.shape[0] if isinstance(sp, Box) else sp.n
                for sp in env.action_space]

    # ================================================================
    # 2. 저장 디렉토리 설정
    # ================================================================
    model_dir = Path('./models') / 'ev_charging' / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst = [int(str(f.name).split('run')[1])
                for f in model_dir.iterdir() if f.name.startswith('run')]
        run_num = max(exst) + 1 if exst else 1

    run_dir  = model_dir / f'run{run_num}'
    log_dir  = run_dir / 'logs'
    plot_dir = run_dir / 'plots'
    os.makedirs(log_dir)
    os.makedirs(plot_dir)
    print(f"  Save dir: {run_dir}\n")

    logger = SummaryWriter(str(log_dir))

    # ================================================================
    # 3. 모델 및 버퍼 초기화
    # ================================================================
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = AttentionSACCMDP.init_from_env(
        env,
        tau=config.tau,
        pi_lr=config.pi_lr,
        q_lr=config.q_lr,
        gamma=config.gamma,
        pol_hidden_dim=config.pol_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        attend_heads=config.attend_heads,
        reward_scale=config.reward_scale,
        n_constraints=n_constraints,
        constraint_thresholds=constraint_thresholds,
        lambda_lr=config.lambda_lr,
        lambda_max=config.lambda_max,
    )

    replay_buffer = CMDPReplayBuffer(
        max_steps=config.buffer_length,
        num_agents=n_agents,
        obs_dims=obs_dims,
        ac_dims=ac_dims,
        n_constraints=n_constraints,
    )

    print(f"  Agents    : {n_agents}")
    print(f"  Obs dim   : {obs_dims[0]}")
    print(f"  Action dim: {ac_dims[0]}")
    print()

    # ================================================================
    # 4. 학습 루프
    # ================================================================
    t = 0
    start_time = time.time()
    best_reward = -float('inf')

    all_rewards    = []
    all_q_losses   = []
    all_qc0_losses = []
    all_pol_losses = [[] for _ in range(n_agents)]
    all_lambda0    = []

    for ep_i in range(config.n_episodes):
        obs_list = env.reset()
        model.prep_rollouts(device='cpu')

        ep_reward     = np.zeros(n_agents)
        ep_cost_c1    = np.zeros(n_agents)   # 과부하 연속 비용 누계
        ep_cost_c2    = np.zeros(n_agents)   # 강제출차 비용 누계
        ep_overload_steps = 0                # 과부하 발생 스텝 수 (모니터링용)
        ep_q_losses   = []
        ep_qc0_losses = []
        ep_qc1_losses = []
        ep_pol_losses = [[] for _ in range(n_agents)]

        for et_i in range(config.episode_length):
            # ── 행동 선택 ───────────────────────────────────────────
            torch_obs = [
                Variable(torch.Tensor(obs_list[i]).unsqueeze(0), requires_grad=False)
                for i in range(n_agents)
            ]
            torch_agent_actions = model.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy()[0] for ac in torch_agent_actions]

            # ── 환경 스텝 ───────────────────────────────────────────
            next_obs_list, rewards, dones, infos = env.step(agent_actions)

            # ── 보상 분리 ────────────────────────────────────────────
            overload_per_agent    = np.array([infos[i].get('overload',    0.0)
                                              for i in range(n_agents)], dtype=np.float32)
            undercharge_per_agent = np.array([infos[i].get('undercharge', 0.0)
                                              for i in range(n_agents)], dtype=np.float32)
            actual_powers         = np.array([infos[i].get('actual_power', 0.0)
                                              for i in range(n_agents)], dtype=np.float32)

            rewards_arr  = np.array(rewards, dtype=np.float32)
            rewards_corr = rewards_arr + overload_per_agent   # C1: 과부하 항상 제거
            if config.d2 is not None:
                rewards_corr = rewards_corr + undercharge_per_agent  # C2: 강제출차도 제거

            # ── C1: 연속 과부하 비용 (p4 직접 사용) ────────────────────────
            # cost_c1[i] = overload_per_agent[i] = p4
            #            = agent_share × P_OVERLOAD × overload_kw × Δt
            # 초과량에 비례하는 실제 페널티 값 → Q_c가 의미 있는 값 학습 가능
            total_power = np.sum(actual_powers)
            overload_kw = max(0.0, total_power - GRID_LIMIT_KW)

            if overload_kw > 0.0:
                cost_c1 = overload_per_agent.copy()  # p4 값 그대로
                ep_overload_steps += 1
            else:
                cost_c1 = np.zeros(n_agents, dtype=np.float32)

            # ── C2: 강제출차 이진 비용 (d2 설정 시) ────────────────────
            cost_c2 = (undercharge_per_agent != 0.0).astype(np.float32)

            # ── 버퍼 저장 ─────────────────────────────────────────────
            obs_arr      = np.array([obs_list],      dtype=np.float32)
            next_obs_arr = np.array([next_obs_list], dtype=np.float32)
            rews_arr2    = rewards_corr[np.newaxis, :]
            dones_arr    = np.array([dones],         dtype=np.uint8)
            ac_buf       = [agent_actions[i][np.newaxis, :] for i in range(n_agents)]
            if config.d2 is None:
                costs_buf = [cost_c1[np.newaxis, :]]
            else:
                costs_buf = [cost_c1[np.newaxis, :], cost_c2[np.newaxis, :]]

            replay_buffer.push(
                obs_arr, ac_buf, rews_arr2, next_obs_arr, dones_arr, costs_buf
            )

            # 통계 누적
            obs_list    = next_obs_list
            t          += 1
            ep_reward  += rewards_corr
            ep_cost_c1 += cost_c1
            ep_cost_c2 += cost_c2

            # ── 모델 업데이트 ────────────────────────────────────────
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) == 0):

                device = 'gpu' if config.use_gpu else 'cpu'
                model.prep_training(device=device)

                for _ in range(config.num_updates):
                    sample = replay_buffer.sample(
                        config.batch_size, to_gpu=config.use_gpu)

                    sample_5 = sample[:5]
                    q_loss = model.update_critic(sample_5, logger=None)

                    qc_losses = model.update_constraint_critics(sample, logger=None)
                    pol_losses = model.update_policies(sample, logger=None)
                    model.update_all_targets()

                    ep_q_losses.append(q_loss)
                    if len(qc_losses) > 0:
                        ep_qc0_losses.append(qc_losses[0])
                    if len(qc_losses) > 1:
                        ep_qc1_losses.append(qc_losses[1])
                    if pol_losses:
                        for a_i, pl in enumerate(pol_losses):
                            ep_pol_losses[a_i].append(pl)

                model.prep_rollouts(device='cpu')

        # ── λ 업데이트: 현재 에피소드 실제 비용 기준 (버퍼 샘플 X) ──────
        ep_mean_c1 = ep_cost_c1 / config.episode_length  # 스텝당 평균
        ep_costs_for_lambda = [
            [torch.tensor(ep_mean_c1[i:i+1]) for i in range(n_agents)]
        ]
        if config.d2 is not None:
            ep_mean_c2 = ep_cost_c2 / config.episode_length
            ep_costs_for_lambda.append(
                [torch.tensor(ep_mean_c2[i:i+1]) for i in range(n_agents)]
            )
        model.update_lambdas(ep_costs_for_lambda)

        # ── 에피소드 종료 처리 ──────────────────────────────────────
        mean_ep_reward = float(np.mean(ep_reward))
        mean_c1 = float(np.mean(ep_cost_c1))   # 에피소드 평균 연속 비용
        mean_c2 = float(np.mean(ep_cost_c2))
        lam0 = model.lambdas[0].item()
        lam1 = model.lambdas[1].item() if n_constraints > 1 else 0.0

        all_rewards.append(mean_ep_reward)
        all_lambda0.append(lam0)
        if ep_q_losses:
            all_q_losses.append(np.mean(ep_q_losses))
        if ep_qc0_losses:
            all_qc0_losses.append(np.mean(ep_qc0_losses))
        for a_i in range(n_agents):
            if ep_pol_losses[a_i]:
                all_pol_losses[a_i].append(np.mean(ep_pol_losses[a_i]))

        # TensorBoard 로깅
        logger.add_scalar('all/mean_episode_reward',              mean_ep_reward,    ep_i)
        logger.add_scalar('constraints/mean_cost_c1_overload',    mean_c1,           ep_i)
        logger.add_scalar('constraints/overload_steps',           ep_overload_steps, ep_i)
        logger.add_scalar('lambdas/lambda1_overload',             lam0,              ep_i)
        if n_constraints > 1:
            logger.add_scalar('constraints/mean_cost_c2_forced',  mean_c2,           ep_i)
            logger.add_scalar('lambdas/lambda2_forced',           lam1,              ep_i)
        for a_i in range(n_agents):
            logger.add_scalar(f'agent{a_i}/mean_episode_rewards', ep_reward[a_i],   ep_i)
        if ep_q_losses:
            logger.add_scalar('losses/q_loss', np.mean(ep_q_losses), ep_i)
        if ep_qc0_losses:
            logger.add_scalar('losses/qc0_loss', np.mean(ep_qc0_losses), ep_i)
        if ep_qc1_losses:
            logger.add_scalar('losses/qc1_loss', np.mean(ep_qc1_losses), ep_i)
        for a_i in range(n_agents):
            if ep_pol_losses[a_i]:
                logger.add_scalar(f'agent{a_i}/losses/pol_loss',
                                  np.mean(ep_pol_losses[a_i]), ep_i)

        # policy entropy 로깅
        with torch.no_grad():
            for a_i in range(n_agents):
                ob = Variable(torch.Tensor(obs_list[a_i]).unsqueeze(0),
                              requires_grad=False)
                _, _, _, _, ent = model.policies[a_i](
                    ob, return_all_probs=True, return_log_pi=True,
                    regularize=True, return_entropy=True)
                logger.add_scalar(f'agent{a_i}/policy_entropy', ent, ep_i)

        # Best 모델 저장
        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            model.prep_rollouts(device='cpu')
            model.save(run_dir / 'model_best.pt')

        # 진행 상황 출력
        elapsed = time.time() - start_time
        if ep_i % config.print_interval == 0 or ep_i == config.n_episodes - 1:
            eta = elapsed / (ep_i + 1) * (config.n_episodes - ep_i - 1)
            c2_str = f" C2(FD):{mean_c2:6.3f} λ2:{lam1:.4f} |" if n_constraints > 1 else ""
            print(
                f"  Ep {ep_i+1:5d}/{config.n_episodes} | "
                f"Rew: {mean_ep_reward:9.1f} | "
                f"Best: {best_reward:9.1f} | "
                f"C1(OL):{mean_c1:.4f} OLstep:{ep_overload_steps:3d} λ1:{lam0:.4f} |"
                f"{c2_str}"
                f" Time: {format_time(elapsed)} ETA: {format_time(eta)}",
                flush=True
            )

        # 체크포인트 저장
        if (ep_i + 1) % config.save_interval == 0:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / f'model_ep{ep_i+1}.pt')
            model.save(run_dir / 'model.pt')

    # ── 최종 저장 ───────────────────────────────────────────────────
    model.prep_rollouts(device='cpu')
    model.save(run_dir / 'model.pt')
    if hasattr(env, 'close'):
        env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

    total_time = time.time() - start_time
    print(f"\n  Training complete! ({format_time(total_time)})")
    print(f"  Model saved : {run_dir / 'model.pt'}")
    print(f"  Best reward : {best_reward:.1f}")
    print(f"  Final λ1    : {model.lambdas[0].item():.4f}")
    if n_constraints > 1:
        print(f"  Final λ2    : {model.lambdas[1].item():.4f}")

    return (str(run_dir), all_q_losses, all_qc0_losses,
            all_pol_losses, all_lambda0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EV Charging CMDP-MAAC Training (Continuous Cost)')

    # 기본 설정
    parser.add_argument('--model_name',   default='cmdp_cont', type=str)
    parser.add_argument('--seed',         default=1,    type=int)
    parser.add_argument('--arrival_mode', default='normal_10', type=str,
                        choices=['normal_10', 'extreme', 'smooth', 'low'])

    # 학습 설정
    parser.add_argument('--n_episodes',      default=20000, type=int)
    parser.add_argument('--episode_length',  default=144,   type=int)
    parser.add_argument('--batch_size',      default=1024,  type=int)
    parser.add_argument('--buffer_length',   default=int(1e6), type=int)
    parser.add_argument('--steps_per_update',default=144,   type=int)
    parser.add_argument('--num_updates',     default=4,     type=int)

    # 모델 하이퍼파라미터
    parser.add_argument('--pol_hidden_dim',    default=128,    type=int)
    parser.add_argument('--critic_hidden_dim', default=128,    type=int)
    parser.add_argument('--attend_heads',      default=4,      type=int)
    parser.add_argument('--pi_lr',             default=0.001,  type=float)
    parser.add_argument('--q_lr',              default=0.0005, type=float)
    parser.add_argument('--tau',               default=0.001,  type=float)
    parser.add_argument('--gamma',             default=0.99,   type=float)
    parser.add_argument('--reward_scale',      default=10.0,   type=float)

    # CMDP 전용 파라미터
    parser.add_argument('--d1', default=0.0, type=float,
                        help='C1(과부하) 허용 임계값. '
                             '단위: 스텝당 평균 (초과kW/gridLimit) 비율. '
                             '0.0 = 완전 금지')
    parser.add_argument('--d2', default=None, type=float,
                        help='C2(강제출차) 허용 임계값. 설정 시 n_constraints=2. '
                             '0.0 = 완전 금지')
    parser.add_argument('--lambda_lr',  default=0.1,   type=float)
    parser.add_argument('--lambda_max', default=10.0,  type=float)

    # 기타
    parser.add_argument('--use_gpu',        action='store_true')
    parser.add_argument('--save_interval',  default=1000, type=int)
    parser.add_argument('--print_interval', default=50,   type=int)

    config = parser.parse_args()

    result = train(config)
    run_dir, q_losses, qc0_losses, pol_losses, lam0_list = result

    print("\nDone!")
