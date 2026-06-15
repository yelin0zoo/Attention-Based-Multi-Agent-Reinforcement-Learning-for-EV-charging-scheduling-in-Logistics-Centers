"""
Random 벤치마크 알고리즘
- 환경, 보상, 상태 전이, 행동 마스킹 등 모든 것이 MAAC와 동일
- 액션 선택만 무작위 (학습 없음)
- 여러 에피소드를 실행하여 평균 성능을 측정
"""
import sys
import io
import os
import csv
import time
import argparse
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from envs.ev_charging.ev_charging_env import EVChargingEnv, NUM_ACTIONS, EPISODE_LENGTH


def run_random_benchmark(config):
    print("=" * 60)
    print("  Random Benchmark")
    print("=" * 60)
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Seed: {config.seed}")
    print()

    env = EVChargingEnv(num_docks=config.num_docks, arrival_mode=config.arrival_mode)
    env.seed(config.seed)
    np_random = np.random.RandomState(config.seed)

    # 결과 저장 디렉토리
    save_dir = f'models/ev_charging/benchmark_random/{config.arrival_mode}_seed{config.seed}'
    os.makedirs(save_dir, exist_ok=True)

    # CSV 로그 초기화
    # 각 충전기별로 페널티 기록할 csv 파일 만드는 작업
    csv_path = os.path.join(save_dir, 'penalty_log.csv')
    penalty_keys = ['charging_cost', 'dissatisfaction', 'undercharge',
                    'overload', 'waiting', 'overtime']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['episode']
        for a_i in range(env.num_docks):
            for key in penalty_keys:
                header.append(f'agent{a_i}_{key}')
            header.append(f'agent{a_i}_total_reward')
        header.append('mean_reward')
        writer.writerow(header)

    # 에피소드 실행
    all_rewards = []
    all_penalties = {key: [] for key in penalty_keys}
    start_time = time.time()

    for ep in range(config.n_episodes):
        obs = env.reset()
        ep_reward = np.zeros(env.num_docks)

        for step in range(EPISODE_LENGTH):
            # 핵심: 액션을 무작위로 선택 (0~9 중 랜덤)
            actions = [np_random.randint(0, NUM_ACTIONS) for _ in range(env.num_docks)]
            obs, rewards, dones, infos = env.step(actions)
            ep_reward += rewards

        # 에피소드 결과 기록
        mean_reward = np.mean(ep_reward)
        all_rewards.append(mean_reward)

        # 페널티 집계
        ep_penalty_totals = {key: 0.0 for key in penalty_keys}
        row = [ep + 1]
        for a_i in range(env.num_docks):
            for key in penalty_keys:
                total = sum(step_info[a_i][key] for step_info in env.ep_penalties)
                row.append(round(total, 4))
                ep_penalty_totals[key] += total
            row.append(round(ep_reward[a_i], 4))
        row.append(round(mean_reward, 4))

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)

        for key in penalty_keys:
            all_penalties[key].append(ep_penalty_totals[key] / env.num_docks)

        # 진행 상황 출력
        if (ep + 1) % config.print_interval == 0 or ep == config.n_episodes - 1:
            elapsed = time.time() - start_time
            recent = all_rewards[-min(50, len(all_rewards)):]
            print(f"  Episode {ep+1:5d}/{config.n_episodes} | "
                  f"Reward: {mean_reward:10.1f} | "
                  f"Avg(50): {np.mean(recent):10.1f} | "
                  f"Time: {elapsed:.0f}s", flush=True)

    # 최종 결과 요약
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  Episodes:      {config.n_episodes}")
    print(f"  Mean reward:   {np.mean(all_rewards):,.1f}")
    print(f"  Std reward:    {np.std(all_rewards):,.1f}")
    print(f"  Min reward:    {np.min(all_rewards):,.1f}")
    print(f"  Max reward:    {np.max(all_rewards):,.1f}")
    print()
    print("  Mean penalties per agent per episode:")
    for key in penalty_keys:
        print(f"    {key:20s}: {np.mean(all_penalties[key]):>12,.1f}")
    print()
    print(f"  Time: {elapsed:.1f}s")
    print(f"  CSV saved: {csv_path}")

    # 요약 통계 저장
    summary_path = os.path.join(save_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Random Benchmark Summary\n")
        f.write(f"Episodes: {config.n_episodes}, Seed: {config.seed}\n\n")
        f.write(f"Mean reward: {np.mean(all_rewards):.1f}\n")
        f.write(f"Std reward:  {np.std(all_rewards):.1f}\n")
        f.write(f"Min reward:  {np.min(all_rewards):.1f}\n")
        f.write(f"Max reward:  {np.max(all_rewards):.1f}\n\n")
        f.write(f"Mean penalties per agent per episode:\n")
        for key in penalty_keys:
            f.write(f"  {key:20s}: {np.mean(all_penalties[key]):>12.1f}\n")

    print(f"  Summary saved: {summary_path}")
    print("=" * 60)

    return all_rewards, all_penalties


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Benchmark for EV Charging')
    parser.add_argument('--n_episodes', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--print_interval', default=10, type=int)
    parser.add_argument('--num_docks', default=10, type=int,
                        help='Number of charging docks')
    parser.add_argument('--arrival_mode', default='normal_10', type=str,
                        choices=['normal_10', 'extreme', 'smooth', 'low'],
                        help='Truck arrival rate mode')
    config = parser.parse_args()
    run_random_benchmark(config)
