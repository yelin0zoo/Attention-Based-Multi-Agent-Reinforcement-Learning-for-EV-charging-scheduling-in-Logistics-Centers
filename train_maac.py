"""
EV 충전 MAAC 학습 + 시각화 통합 실행 스크립트

사용법:
  python train.py                        # 기본 설정으로 학습
  python train.py --n_episodes 5000      # 에피소드 수 변경
  python train.py --use_gpu              # GPU 사용

학습 완료 후 자동으로 시각화 그래프가 생성됩니다.
"""
import sys
import io
import os
import time
import argparse
import subprocess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def format_time(seconds):
    """초를 시:분:초 형식으로 변환"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def train(config):
    """MAAC 학습 실행 (main.py의 run() 함수를 직접 호출)"""
    import torch
    import numpy as np
    from pathlib import Path
    from torch.autograd import Variable
    from tensorboardX import SummaryWriter
    from gym.spaces import Box, Discrete

    from utils.make_env import make_env               # 환경 생성용
    from utils.buffer import ReplayBuffer             # 경험 데이터 저장용
    from utils.env_wrappers import DummyVecEnv        # 환경 래퍼용
    from algorithms.attention_sac import AttentionSAC # 모델 학습용

    # ============================================================
    # 1. 환경 생성
    # ============================================================
    print("=" * 60)
    print("  EV Charging MAAC Training")
    print("=" * 60)
    print(f"\n  Episodes:       {config.n_episodes}")
    print(f"  Episode length: {config.episode_length} steps (= 24 hours)")
    print(f"  Reward scale:   {config.reward_scale}")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  GPU:            {config.use_gpu}")
    print(f"  Arrival mode:   {config.arrival_mode}")
    print()

    # 환경 생성 함수를 반환하는 함수
    def get_env_fn(seed):
        def init_env():
            env = make_env('ev_charging', discrete_action=True,
                           arrival_mode=config.arrival_mode)
            env.seed(seed)
            return env
        return init_env
    
    # 환경을 만드는 함수
    env = DummyVecEnv([get_env_fn(config.seed)])

    # ============================================================
    # 2. 모델 저장 디렉토리
    # ============================================================
    model_dir = Path('./models') / 'ev_charging' / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst = [int(str(f.name).split('run')[1])
                for f in model_dir.iterdir() if f.name.startswith('run')]
        run_num = max(exst) + 1 if exst else 1

    run_dir = model_dir / f'run{run_num}'
    log_dir = run_dir / 'logs'
    plot_dir = run_dir / 'plots'
    os.makedirs(log_dir)
    os.makedirs(plot_dir)
    print(f"  Save dir: {run_dir}\n")

    logger = SummaryWriter(str(log_dir))

    # ============================================================
    # 3. 모델 및 버퍼 초기화
    # ============================================================
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = AttentionSAC.init_from_env(
        env,
        tau=config.tau,
        pi_lr=config.pi_lr,
        q_lr=config.q_lr,
        gamma=config.gamma,
        pol_hidden_dim=config.pol_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        attend_heads=config.attend_heads,
        reward_scale=config.reward_scale,
    )

    replay_buffer = ReplayBuffer(
        config.buffer_length, model.nagents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
         for acsp in env.action_space]
    )

    print(f"  Agents:     {model.nagents}")
    print(f"  Obs dim:    {env.observation_space[0].shape[0]}")
    print(f"  Action dim: {env.action_space[0].n}")
    print()

    # ============================================================
    # 4. 학습 루프
    # ============================================================
    t = 0
    start_time = time.time()
    best_reward = -float('inf')
    all_rewards = []
    all_q_losses = []
    all_pol_losses = [[] for _ in range(model.nagents)]

    for ep_i in range(config.n_episodes):
        obs = env.reset()
        model.prep_rollouts(device='cpu')
        ep_reward = np.zeros(model.nagents)
        ep_q_losses = []
        ep_pol_losses = [[] for _ in range(model.nagents)]

        for et_i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            torch_agent_actions = model.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(1)]

            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += 1
            ep_reward += rewards[0]

            # 모델 업데이트
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) == 0):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for _ in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    # logger=None → 에피소드 단위로 직접 로깅 (x축 ep_i 통일)
                    q_loss = model.update_critic(sample, logger=None)
                    pol_loss = model.update_policies(sample, logger=None)
                    model.update_all_targets()
                    ep_q_losses.append(q_loss)
                    if pol_loss:
                        for a_i, pl in enumerate(pol_loss):
                            ep_pol_losses[a_i].append(pl)
                model.prep_rollouts(device='cpu')

        # loss 기록
        all_rewards.append(np.mean(ep_reward))
        if ep_q_losses:
            all_q_losses.append(np.mean(ep_q_losses))
        for a_i in range(model.nagents):
            if ep_pol_losses[a_i]:
                all_pol_losses[a_i].append(np.mean(ep_pol_losses[a_i]))

        # 에피소드 단위 로깅 (x축 ep_i 기준으로 통일)
        mean_ep_reward = np.mean(ep_reward)
        for a_i in range(model.nagents):
            logger.add_scalar(f'agent{a_i}/mean_episode_rewards',
                              ep_reward[a_i], ep_i)
        logger.add_scalar('all/mean_episode_reward', mean_ep_reward, ep_i)
        if ep_q_losses:
            logger.add_scalar('losses/q_loss', np.mean(ep_q_losses), ep_i)
        for a_i in range(model.nagents):
            if ep_pol_losses[a_i]:
                logger.add_scalar(f'agent{a_i}/losses/pol_loss',
                                  np.mean(ep_pol_losses[a_i]), ep_i)

        # policy entropy 로깅 (에피소드 마지막 obs 기준)
        with torch.no_grad():
            for a_i in range(model.nagents):
                ob = Variable(torch.Tensor(obs[0, a_i]).unsqueeze(0),
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
            print(f"  Episode {ep_i+1:5d}/{config.n_episodes} | "
                  f"Reward: {mean_ep_reward:9.1f} | "
                  f"Best: {best_reward:9.1f} | "
                  f"Time: {format_time(elapsed)} | "
                  f"ETA: {format_time(eta)}", flush=True)

        # 모델 체크포인트 저장
        if (ep_i + 1) % config.save_interval == 0:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / f'model_ep{ep_i+1}.pt')
            model.save(run_dir / 'model.pt')

    # 최종 저장
    model.prep_rollouts(device='cpu')
    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

    total_time = time.time() - start_time
    print(f"\n  Training complete! ({format_time(total_time)})")
    print(f"  Model saved: {run_dir / 'model.pt'}")
    print(f"  Best reward: {best_reward:.1f}")

    return str(run_dir), all_q_losses, all_pol_losses


#  학습 결과 시각화
def visualize(run_dir):
    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60 + "\n")

    from visualize import (
        load_tb_rewards, plot_reward_curves,
        run_eval_episode, print_summary,
        plot_soc, plot_power_heatmap, plot_rewards_breakdown
    )
    from pathlib import Path

    run_dir = Path(run_dir)
    plot_dir = run_dir / 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # 1. 보상 곡선
    log_dir = run_dir / 'logs'
    if log_dir.exists():
        print("[1/4] Reward curves...")
        agent_rewards = load_tb_rewards(log_dir)
        if agent_rewards:
            plot_reward_curves(agent_rewards, plot_dir / '1_reward_curves.png')

    # 2. 평가 에피소드 (학습된 모델)
    model_path = run_dir / 'model.pt'
    print("[2/4] Evaluation episode...")
    data = run_eval_episode(
        model_path=str(model_path) if model_path.exists() else None,
        seed=42
    )
    print_summary(data)

    # 3. SoC 그래프
    print("[3/4] SoC trajectories...")
    plot_soc(data, plot_dir / '2_soc_trajectories.png')

    # 4. 전력 + 보상 그래프
    print("[4/4] Power & rewards...")
    plot_power_heatmap(data, plot_dir / '3_power_heatmap.png')
    plot_rewards_breakdown(data, plot_dir / '4_rewards_breakdown.png')

    print(f"\n  All plots saved to: {plot_dir}")


def plot_loss_curves(run_dir, all_q_losses, all_pol_losses):
    """Loss 곡선 그래프 저장"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    run_dir = Path(run_dir)
    plot_dir = run_dir / 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Critic Loss
    if all_q_losses:
        ax1.plot(all_q_losses, alpha=0.4, color='gray')
        window = min(50, len(all_q_losses))
        smoothed = np.convolve(all_q_losses, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, color='red', linewidth=2)
    ax1.set_title('Critic Loss (q_loss)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # Policy Loss
    colors = ['blue', 'orange', 'green', 'purple']
    for a_i, (losses, color) in enumerate(zip(all_pol_losses, colors)):
        if losses:
            ax2.plot(losses, alpha=0.3, color=color)
            window = min(50, len(losses))
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, color=color, linewidth=2, label=f'Agent {a_i}')
    ax2.set_title('Policy Loss (pol_loss)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plot_dir / '5_loss_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EV Charging MAAC Train + Visualize')

    # 기본 설정
    parser.add_argument('--model_name', default='my_model', type=str,
                        help='Model directory name')
    parser.add_argument('--seed', default=1, type=int)

    # 학습 설정
    parser.add_argument('--n_episodes', default=3000, type=int,
                        help='Total training episodes')
    parser.add_argument('--episode_length', default=144, type=int,
                        help='Steps per episode (144 = 24h)')
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--buffer_length', default=int(1e6), type=int)
    parser.add_argument('--steps_per_update', default=144, type=int,
                        help='Update model every N steps')
    parser.add_argument('--num_updates', default=4, type=int)

    # 모델 설정
    parser.add_argument('--pol_hidden_dim', default=128, type=int)
    parser.add_argument('--critic_hidden_dim', default=128, type=int)
    parser.add_argument('--attend_heads', default=4, type=int)
    parser.add_argument('--pi_lr', default=0.001, type=float)
    parser.add_argument('--q_lr', default=0.0005, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=10.0, type=float)

    # 기타
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--print_interval', default=50, type=int,
                        help='Print progress every N episodes')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip visualization after training')
    parser.add_argument('--arrival_mode', default='normal', type=str,
                        choices=['normal', 'extreme', 'smooth', 'low', 'normal_10'],
                        help='Truck arrival rate mode')

    config = parser.parse_args()

    # 학습 실행
    run_dir, all_q_losses, all_pol_losses = train(config)

    # 시각화 실행
    if not config.no_viz:
        visualize(run_dir)
        plot_loss_curves(run_dir, all_q_losses, all_pol_losses)

    print("\nDone!")
