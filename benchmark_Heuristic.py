"""
Heuristic 벤치마크 알고리즘
- 환경, 보상, 상태 전이, 행동 마스킹 등 모든 것이 MAAC와 동일
- 액션 선택만 긴급도 + 전기 요금 기반 규칙 (학습 없음)

규칙 (우선순위):
  1순위 - 긴급 충전: 마감 임박 + SoC 부족 → 최대 전력 (90kW)
  2순위 - 마무리 충전: SoC가 목표에 거의 도달 → 최소 전력 (10kW)
  3순위 - 요금 기반:
    경부하 (79.2원, 22:00~08:00)    → 90kW
    중간부하 (137.4원)               → 50kW
    최대부하 (190.4원, 11~12,13~18) → 10kW

  → 환경의 action masking이 빈 도크, 충전 완료+적재 중 등을
    자동으로 0kW로 강제하므로 별도 예외 처리 불필요
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

from envs.ev_charging.ev_charging_env import (
    EVChargingEnv, NUM_ACTIONS, ACTION_KW, EPISODE_LENGTH, DELTA_T
)
from envs.ev_charging.price_schedule import get_price_schedule


def urgency_price_action(dock, price):
    """
    긴급도 + 전기 요금에 따라 충전 전력(액션 인덱스)을 결정하는 규칙

    ACTION_KW = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] kW
    인덱스:       0   1   2   3   4   5   6   7   8   9

    인자:
        dock: 도크 상태 dict
        price: 현재 전기 요금 (원/kWh)
    반환:
        int: 액션 인덱스 (0~9)
    """
    # 빈 도크는 환경이 action masking으로 처리하지만, 여기서도 0 반환
    if dock['connected'] == 0:
        return 0

    soc = dock['soc']
    target = dock['target_soc']
    remain = dock['departure_remain']
    soc_gap = target - soc  # 남은 충전량 (0~1)

    # --- 1순위: 긴급 충전 ---
    # 남은 시간 대비 충전이 부족한 경우 최대 전력
    # 필요 에너지(kWh) = soc_gap × battery_cap
    # 필요 시간(스텝) = 필요 에너지 / (max_power × DELTA_T)
    if soc_gap > 0 and dock['battery_cap'] > 0:
        max_power = min(dock['charger_max_kw'], dock['ev_max_kw'])
        if max_power > 0:
            needed_steps = (soc_gap * dock['battery_cap']) / (max_power * DELTA_T)
            # 남은 시간이 필요 시간의 1.5배 이하면 긴급
            if remain <= needed_steps * 1.5:
                return 9  # 90kW

    # --- 2순위: 충전 거의 완료 ---
    # 목표까지 5% 이하 남았으면 최소 전력으로 마무리
    if soc_gap <= 0.05:
        return 1  # 10kW

    # --- 3순위: 요금 기반 ---
    if price <= 79.2:       # 경부하
        return 9            # 90kW
    elif price <= 137.4:    # 중간부하
        return 5            # 50kW
    else:                   # 최대부하
        return 1            # 10kW


def run_urgency_price_benchmark(config):
    print("=" * 60)
    print("  Urgency-Price Rule Benchmark")
    print("=" * 60)
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Seed: {config.seed}")
    print()
    print("  Rule (priority order):")
    print("    1. Urgent (deadline close + SoC low) → 90kW")
    print("    2. Almost done (SoC gap <= 5%)       → 10kW")
    print("    3. Off-peak (79.2won)                → 90kW")
    print("    4. Mid-peak (137.4won)               → 50kW")
    print("    5. Peak (190.4won)                   → 10kW")
    print()

    env = EVChargingEnv(num_docks=config.num_docks, arrival_mode=config.arrival_mode)
    env.seed(config.seed)

    price_schedule = get_price_schedule()

    # 결과 저장 디렉토리
    save_dir = f'models/ev_charging/benchmark_Heuristic/{config.arrival_mode}_seed{config.seed}'
    os.makedirs(save_dir, exist_ok=True)

    # CSV 로그 초기화
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
            step_idx = min(step, EPISODE_LENGTH - 1)
            current_price = price_schedule[step_idx]

            # 핵심: 각 도크의 상태를 보고 개별적으로 액션 결정
            actions = []
            for i in range(env.num_docks):
                action_idx = urgency_price_action(env.dock_states[i], current_price)
                actions.append(action_idx)

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
        f.write(f"Heuristic Benchmark Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Episodes: {config.n_episodes}, Seed: {config.seed}\n\n")
        f.write(f"Rule (priority order):\n")
        f.write(f"  1. Urgent (deadline close + SoC low) -> 90kW\n")
        f.write(f"  2. Almost done (SoC gap <= 5%)       -> 10kW\n")
        f.write(f"  3. Off-peak (79.2won)                -> 90kW\n")
        f.write(f"  4. Mid-peak (137.4won)               -> 50kW\n")
        f.write(f"  5. Peak (190.4won)                   -> 10kW\n\n")
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
    parser = argparse.ArgumentParser(
        description='Urgency-Price Rule Benchmark for EV Charging'
    )
    parser.add_argument('--n_episodes', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--print_interval', default=10, type=int)
    parser.add_argument('--num_docks', default=10, type=int,
                        help='Number of charging docks')
    parser.add_argument('--arrival_mode', default='normal_10', type=str,
                        choices=['normal_10', 'extreme', 'smooth', 'low'],
                        help='Truck arrival rate mode')
    config = parser.parse_args()
    run_urgency_price_benchmark(config)
