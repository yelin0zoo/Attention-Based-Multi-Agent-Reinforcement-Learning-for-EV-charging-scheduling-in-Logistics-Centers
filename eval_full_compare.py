# -*- coding: utf-8 -*-
"""
eval_full_compare.py — 종합 비교 평가 (그림 각각 1개씩 저장)

생성 그래프:
  0_training_curves/    : SAC / MAAC / C-MAAC 학습 곡선 (TensorBoard 로그 기반, 1장)
  1_baseline_s1/        : Random/Heuristic/SAC/MAAC/C-MAAC, S1 기준
                           - Total Penalty, 6개 페널티 항목, Overload Steps,
                             Forced Departures  →  각각 별도 PNG (총 9장)
  2_sac_vs_maac_s1/      : SAC vs MAAC, S1
                           - Total Penalty, Overload, Forced Dep (3장)
  3_sac_vs_maac_scen/    : SAC vs MAAC, 시나리오별(S1~S10)
                           - Total Penalty, Overload, Forced Dep (3장)
  4_maac_vs_cmdp_s1/     : MAAC vs C-MAAC, S1
                           - Total Penalty, Overload, Forced Dep (3장)
  5_maac_vs_cmdp_scen/   : MAAC vs C-MAAC, 시나리오별(S1~S10)
                           - Total Penalty, Overload, Forced Dep (3장)

사용법:
  python eval_full_compare.py \
      --maac_dirs models/ev_charging/maac_v2_s1/run1 \
                  models/ev_charging/maac_v2_s2/run1 \
                  models/ev_charging/maac_v2_s3/run1 \
      --sac_dirs  models/ev_charging/sac_v1_s1/run1 \
                  models/ev_charging/sac_v1_s2/run1 \
                  models/ev_charging/sac_v1_s3/run1 \
      --cmdp_dirs models/ev_charging/cmdp_v9_s1/run1 \
                  models/ev_charging/cmdp_v9_s2/run1 \
                  models/ev_charging/cmdp_v9_s3/run1 \
      --n_seeds 50 --out_dir full_compare
"""

import sys, io, os, argparse
import numpy as np
import torch
from pathlib import Path
from torch.autograd import Variable

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import envs.ev_charging.ev_charging_env as _env_mod
from envs.ev_charging.ev_charging_env import (
    EVChargingEnv, EPISODE_LENGTH, NUM_ACTIONS, DELTA_T)
from envs.ev_charging.price_schedule import get_price_schedule
from utils.single_wrapper import MultiToSingleWrapper

N_AGENTS = 10

PENALTY_KEYS   = ['charging_cost', 'dissatisfaction', 'undercharge',
                  'overload', 'waiting', 'overtime']
PENALTY_LABELS = ['Charging Cost', 'Dissatisfaction', 'Undercharge',
                  'Overload', 'Waiting', 'Overtime']

SCENARIOS = [
    dict(name='S1',  label='S1\n500kW\n(base)',    grid_limit=500, arrival_mode='normal_10', num_failed=0, battery_cap=60.0),
    dict(name='S2',  label='S2\n400kW',            grid_limit=400, arrival_mode='normal_10', num_failed=0, battery_cap=60.0),
    dict(name='S3',  label='S3\n300kW',            grid_limit=300, arrival_mode='normal_10', num_failed=0, battery_cap=60.0),
    dict(name='S4',  label='S4\nExtreme',          grid_limit=500, arrival_mode='extreme',    num_failed=0, battery_cap=60.0),
    dict(name='S5',  label='S5\n400kW\n+Extreme',  grid_limit=400, arrival_mode='extreme',    num_failed=0, battery_cap=60.0),
    dict(name='S6',  label='S6\nFail x1\n(9dock)', grid_limit=500, arrival_mode='normal_10', num_failed=1, battery_cap=60.0),
    dict(name='S7',  label='S7\nFail x2\n(8dock)', grid_limit=500, arrival_mode='normal_10', num_failed=2, battery_cap=60.0),
    dict(name='S8',  label='S8\nFail x3\n(7dock)', grid_limit=500, arrival_mode='normal_10', num_failed=3, battery_cap=60.0),
    dict(name='S9',  label='S9\nBattery\n50kWh',   grid_limit=500, arrival_mode='normal_10', num_failed=0, battery_cap=50.0),
    dict(name='S10', label='S10\nBattery\n70kWh',  grid_limit=500, arrival_mode='normal_10', num_failed=0, battery_cap=70.0),
]


# ── 시나리오 환경 ─────────────────────────────────────────────────────────
class ScenarioEnv(EVChargingEnv):
    def __init__(self, scenario, num_docks=10):
        super().__init__(num_docks=num_docks, arrival_mode=scenario['arrival_mode'])
        self.grid_limit_kw = float(scenario['grid_limit'])
        self.failed_docks  = set(range(scenario['num_failed']))
        self.battery_cap   = float(scenario.get('battery_cap', 60.0))

    def reset(self):
        import envs.ev_charging.truck_generator as _tg
        orig_grid = _env_mod.GRID_LIMIT_KW
        orig_bat  = _tg.BATTERY_CAP_KWH
        _env_mod.GRID_LIMIT_KW = self.grid_limit_kw
        _tg.BATTERY_CAP_KWH    = self.battery_cap
        result = super().reset()
        _env_mod.GRID_LIMIT_KW = orig_grid
        _tg.BATTERY_CAP_KWH    = orig_bat
        return result

    def step(self, actions):
        orig = _env_mod.GRID_LIMIT_KW
        _env_mod.GRID_LIMIT_KW = self.grid_limit_kw
        result = super().step(actions)
        _env_mod.GRID_LIMIT_KW = orig
        return result

    def _process_new_arrivals(self):
        for i in range(self.num_docks):
            if i in self.failed_docks:
                continue
            if self.dock_states[i]['connected'] == 0 and self.arrival_queue:
                truck = self.arrival_queue[0]
                if truck['arrival_step'] <= self.current_step:
                    self.arrival_queue.popleft()
                    self._assign_truck(i, truck)

    def _apply_action_masking(self, i, action_kw):
        if i in self.failed_docks:
            return 0.0
        return super()._apply_action_masking(i, action_kw)


# ── 결과 집계 헬퍼 ────────────────────────────────────────────────────────
def _empty():
    return dict(total_penalty=0.0, overload_steps=0, forced_dep=0,
                 completed_docks=np.zeros(N_AGENTS), forced_docks=np.zeros(N_AGENTS),
                 **{k: 0.0 for k in PENALTY_KEYS})


def _update_dock_completion(env, connected_before, infos, res, n):
    """도크별 정상출차(completed)/강제출차(forced) 카운트 갱신"""
    for i in range(n):
        if connected_before[i] == 1 and env.dock_states[i]['connected'] == 0:
            if isinstance(infos[i], dict) and infos[i].get('undercharge', 0) > 0:
                res['forced_docks'][i] += 1
            else:
                res['completed_docks'][i] += 1


def _collect(infos):
    pens = {k: 0.0 for k in PENALTY_KEYS}
    overload_step = any(info.get('overload', 0) > 0
                        for info in infos if isinstance(info, dict))
    forced_dep = 0
    for info in infos:
        if not isinstance(info, dict):
            continue
        for k in PENALTY_KEYS:
            pens[k] += abs(info.get(k, 0.0))
        if info.get('undercharge', 0) > 0:
            forced_dep += 1
    return pens, int(overload_step), forced_dep


# ── 평가 함수들 ───────────────────────────────────────────────────────────
def eval_random(scenario, seed):
    rng = np.random.RandomState(seed)
    env = ScenarioEnv(scenario, num_docks=N_AGENTS)
    env.seed(seed)
    env.reset()
    res = _empty()
    for _ in range(EPISODE_LENGTH):
        actions = [rng.randint(0, NUM_ACTIONS) for _ in range(N_AGENTS)]
        _, rewards, _, infos = env.step(actions)
        pens, ol, fd = _collect(infos)
        res['total_penalty'] += sum(abs(r) for r in rewards)
        res['overload_steps'] += ol
        res['forced_dep']     += fd
        for k in PENALTY_KEYS:
            res[k] += pens[k]
    return res


def _heuristic_action(dock, price):
    if dock['connected'] == 0:
        return 0
    soc_gap = dock['target_soc'] - dock['soc']
    remain  = dock['departure_remain']
    if soc_gap > 0 and dock['battery_cap'] > 0:
        max_pw = min(dock['charger_max_kw'], dock['ev_max_kw'])
        if max_pw > 0:
            needed = (soc_gap * dock['battery_cap']) / (max_pw * DELTA_T)
            if remain <= needed * 1.5:
                return 9
    if soc_gap <= 0.05:
        return 1
    if price <= 79.2:    return 9
    elif price <= 137.4: return 5
    else:                return 1


def eval_heuristic(scenario, seed):
    env = ScenarioEnv(scenario, num_docks=N_AGENTS)
    env.seed(seed)
    env.reset()
    price_schedule = get_price_schedule()
    res = _empty()
    for step in range(EPISODE_LENGTH):
        price   = price_schedule[min(step, EPISODE_LENGTH - 1)]
        actions = [_heuristic_action(env.dock_states[i], price) for i in range(N_AGENTS)]
        _, rewards, _, infos = env.step(actions)
        pens, ol, fd = _collect(infos)
        res['total_penalty'] += sum(abs(r) for r in rewards)
        res['overload_steps'] += ol
        res['forced_dep']     += fd
        for k in PENALTY_KEYS:
            res[k] += pens[k]
    return res


def eval_maac(model_path, scenario, seed):
    from algorithms.attention_sac import AttentionSAC
    np.random.seed(seed); torch.manual_seed(seed)

    model = AttentionSAC.init_from_save(model_path)
    model.prep_rollouts(device='cpu')
    n = model.nagents

    env = ScenarioEnv(scenario, num_docks=n)
    env.seed(seed)
    obs = env.reset()

    res = _empty()
    for _ in range(EPISODE_LENGTH):
        connected_before = [env.dock_states[i]['connected'] for i in range(n)]
        torch_obs = [Variable(torch.Tensor(obs[i:i+1]), requires_grad=False)
                     for i in range(n)]
        with torch.no_grad():
            acts = model.step(torch_obs, explore=False)
        actions = [int(np.argmax(a.data.numpy()[0])) for a in acts]
        obs, rewards, _, infos = env.step(actions)
        pens, ol, fd = _collect(infos)
        _update_dock_completion(env, connected_before, infos, res, n)
        res['total_penalty'] += sum(abs(r) for r in rewards)
        res['overload_steps'] += ol
        res['forced_dep']     += fd
        for k in PENALTY_KEYS:
            res[k] += pens[k]
    return res


def eval_cmdp(model_path, scenario, seed):
    from algorithms.attention_sac_cmdp import AttentionSACCMDP
    np.random.seed(seed); torch.manual_seed(seed)

    model = AttentionSACCMDP.init_from_save(model_path)
    model.prep_rollouts(device='cpu')
    n = model.nagents

    env = ScenarioEnv(scenario, num_docks=n)
    env.seed(seed)
    obs = env.reset()

    res = _empty()
    for _ in range(EPISODE_LENGTH):
        torch_obs = [Variable(torch.Tensor(obs[i:i+1]), requires_grad=False)
                     for i in range(n)]
        with torch.no_grad():
            acts = model.step(torch_obs, explore=False)
        actions = [int(np.argmax(a.data.numpy()[0])) for a in acts]
        obs, rewards, _, infos = env.step(actions)
        pens, ol, fd = _collect(infos)
        res['total_penalty'] += sum(abs(r) for r in rewards)
        res['overload_steps'] += ol
        res['forced_dep']     += fd
        for k in PENALTY_KEYS:
            res[k] += pens[k]
    return res


def eval_sac(model_path, scenario, seed):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_sac", Path(__file__).parent / "train_sac.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    np.random.seed(seed); torch.manual_seed(seed)
    model = mod.SacDiscreteAgent.init_from_save(model_path)
    model.prep_rollouts(device='cpu')
    n = model.n_agents

    ma_env = ScenarioEnv(scenario, num_docks=n)
    ma_env.seed(seed)
    env = MultiToSingleWrapper(ma_env)
    obs = env.reset()

    res = _empty()
    for _ in range(EPISODE_LENGTH):
        connected_before = [ma_env.dock_states[i]['connected'] for i in range(n)]
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = model.exploit(obs_t)
        obs, reward, _, infos = env.step(action)
        if isinstance(infos, (list, tuple)):
            pens, ol, fd = _collect(infos)
            _update_dock_completion(ma_env, connected_before, infos, res, n)
        else:
            pens, ol, fd = {k: 0.0 for k in PENALTY_KEYS}, 0, 0
        res['total_penalty'] += abs(reward)
        res['overload_steps'] += ol
        res['forced_dep']     += fd
        for k in PENALTY_KEYS:
            res[k] += pens[k]
    return res


# ── 집계 ──────────────────────────────────────────────────────────────────
def aggregate(results):
    keys = results[0].keys()
    out = {}
    for k in keys:
        stacked = np.stack([np.asarray(r[k]) for r in results])
        if stacked.ndim > 1:
            out[k] = (np.mean(stacked, axis=0), np.std(stacked, axis=0))
        else:
            out[k] = (float(np.mean(stacked)), float(np.std(stacked)))
    return out


# ── 단일 막대(모델 N개) 그래프 헬퍼 ───────────────────────────────────────
# 알고리즘별 고유 색상 (bar/face, error-bar/edge, text) — 모든 그래프에서 통일
PALETTE = {
    'Random':    ('#b0bec5', '#78909c', '#37474f'),  # 회색
    'Heuristic': ('#b0bec5', '#78909c', '#37474f'),  # 회색 (Random과 동일)
    'SAC':       ('#b0bec5', '#78909c', '#37474f'),  # 회색 (Random/Heuristic과 동일)
    'MAAC':      ('#90caf9', '#42a5f5', '#1565c0'),  # 파스텔 블루
    'C-MAAC':    ('#ef9a9a', '#e57373', '#c62828'),  # 코랄 (강조)
}


def plot_single_metric_bar(models, vals, stds, title, ylabel, out_path, subtitle='', ylim_max=None):
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(models)), 5))
    x = np.arange(len(models))

    top = ylim_max if ylim_max is not None else max(v + s for v, s in zip(vals, stds)) * 1.18
    top = max(top, 1e-6)

    for i, m in enumerate(models):
        bc, ec, tc = PALETTE.get(m, ('#b0bec5', '#78909c', '#37474f'))
        v, s = vals[i], stds[i]
        ax.bar(i, v, color=bc, alpha=0.88, width=0.55,
               yerr=[[min(s, v)], [s]], capsize=5,
               error_kw=dict(elinewidth=1.3, ecolor=ec, capthick=1.3))
        fmt = f'{v:,.0f}' if abs(v) >= 100 else f'{v:.2f}'
        ax.text(i, v + top * 0.02, fmt,
                ha='center', va='bottom', fontsize=10, color=tc, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    full_title = title + (f'\n({subtitle})' if subtitle else '')
    ax.set_title(full_title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, top)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out_path}")


# ── 2-way 시나리오별 그래프 헬퍼 ───────────────────────────────────────────
def plot_scenario_bar(names, labels, valsA, stdsA, valsB, stdsB, nameA, nameB,
                       title, ylabel, out_path, subtitle='', ylim_max=None):
    x = np.arange(len(names)); w = 0.35
    colA = PALETTE.get(nameA, ('#b0bec5', '#78909c', '#37474f'))
    colB = PALETTE.get(nameB, ('#ef9a9a', '#e57373', '#c62828'))

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), 5.5))

    def safe_yerr(vals, stds):
        return [[min(s, v) for v, s in zip(vals, stds)], list(stds)]

    ax.bar(x - w/2, valsA, w, yerr=safe_yerr(valsA, stdsA), capsize=4,
           label=nameA, color=colA[0], alpha=0.88,
           error_kw=dict(elinewidth=1.2, ecolor=colA[1]))
    ax.bar(x + w/2, valsB, w, yerr=safe_yerr(valsB, stdsB), capsize=4,
           label=nameB, color=colB[0], alpha=0.88,
           error_kw=dict(elinewidth=1.2, ecolor=colB[1]))

    top = ylim_max if ylim_max is not None else max(
        max(v + s for v, s in zip(valsA, stdsA)),
        max(v + s for v, s in zip(valsB, stdsB)),
    ) * 1.18
    top = max(top, 1e-6)

    fmt = ',.0f' if max(valsA + valsB) >= 100 else '.2f'
    bbox = dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.75, edgecolor='none')
    for i, (va, vb) in enumerate(zip(valsA, valsB)):
        ax.text(i - w/2, va + top * 0.02, f'{va:{fmt}}',
                ha='center', fontsize=7, color=colA[2], fontweight='bold', bbox=bbox)
        ax.text(i + w/2, vb + top * 0.02, f'{vb:{fmt}}',
                ha='center', fontsize=7, color=colB[2], fontweight='bold', bbox=bbox)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    full_title = title + (f'\n({subtitle})' if subtitle else '')
    ax.set_title(full_title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, top)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out_path}")


def plot_dock_completion_rate(dock_labels, ratesA, ratesB, nameA, nameB,
                               title, out_path, subtitle=''):
    x = np.arange(len(dock_labels)); w = 0.35
    colA = PALETTE.get(nameA, ('#b0bec5', '#78909c', '#37474f'))
    colB = PALETTE.get(nameB, ('#ef9a9a', '#e57373', '#c62828'))

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(dock_labels)), 5.5))
    ax.bar(x - w/2, ratesA, w, label=nameA, color=colA[0], alpha=0.88)
    ax.bar(x + w/2, ratesB, w, label=nameB, color=colB[0], alpha=0.88)

    for i, (ra, rb) in enumerate(zip(ratesA, ratesB)):
        ax.text(i - w/2, ra + 0.5, f'{ra:.0f}%', ha='center', fontsize=8,
                color=colA[2], fontweight='bold')
        ax.text(i + w/2, rb + 0.5, f'{rb:.0f}%', ha='center', fontsize=8,
                color=colB[2], fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(dock_labels, fontsize=9)
    ax.set_ylabel('Charging Completion Rate (%)', fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='gray', lw=1, ls='--', alpha=0.5)
    full_title = title + (f'\n(% of trucks that reached target SoC before departure | {subtitle})'
                          if subtitle else
                          '\n(% of trucks that reached target SoC before departure)')
    ax.set_title(full_title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out_path}")


# ── 학습 곡선 (TensorBoard) ─────────────────────────────────────────────────
def _load_tb_reward(log_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [오류] pip install tensorboard")
        return None, None

    log_dir = Path(log_dir)
    ea = EventAccumulator(str(log_dir), size_guidance={'scalars': 0})
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    for t in ['all/mean_episode_reward', 'all/mean_episode_rewards', 'mean_episode_reward']:
        if t in tags:
            events = ea.Scalars(t)
            steps  = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            return steps, values

    print(f"    [경고] 태그 없음. 사용 가능: {tags[:10]}")
    return None, None


def _smooth(values, window):
    if len(values) < window:
        window = max(1, len(values) // 5)
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def _load_tb_runs(run_dirs, model_name, scale=1.0):
    print(f"  [{model_name}] 로딩 중...", flush=True)
    runs = []
    for rd in run_dirs:
        log_dir = Path(rd) / 'logs'
        if not log_dir.exists():
            log_dir = Path(rd)
        s, v = _load_tb_reward(log_dir)
        if s is not None:
            runs.append((s, v * scale))

    if not runs:
        print(f"    [경고] {model_name} 데이터 없음")
        return None, None

    min_len = min(len(v) for _, v in runs)
    steps   = runs[0][0][:min_len]
    stacked = np.array([v[:min_len] for _, v in runs])
    return steps, stacked


def _compute_smooth(steps, stacked, window):
    mean_val = np.mean(stacked, axis=0)
    std_val  = np.std(stacked,  axis=0)
    w = min(window, max(1, len(mean_val) // 5))
    if w > 1:
        sm_mean  = _smooth(mean_val, w)
        sm_std   = _smooth(std_val,  w)
        sm_steps = steps[w - 1:]
    else:
        sm_mean, sm_std, sm_steps = mean_val, std_val, steps
    return sm_steps, sm_mean, sm_std


# 학습 곡선도 PALETTE의 (face, txt) 색상을 그대로 사용해 색상 통일
TRAIN_CURVE_STYLES = {
    name: dict(color=face, txt_color=txt)
    for name, (face, _, txt) in PALETTE.items()
}


def plot_training_curves(maac_dirs, sac_dirs, cmdp_dirs, sac_n_agents, out_path,
                          window=300):
    models_data = []
    for name, dirs, scale in [
        ('SAC',    sac_dirs,  1.0 / sac_n_agents),
        ('MAAC',   maac_dirs, 1.0),
        ('C-MAAC', cmdp_dirs, 1.0),
    ]:
        if not dirs:
            continue
        steps, stacked = _load_tb_runs(dirs, name, scale)
        if steps is not None:
            models_data.append({'name': name, 'steps': steps, 'stacked': stacked,
                                **TRAIN_CURVE_STYLES[name]})

    if not models_data:
        print("  [스킵] 학습 로그 없어 training curves 생략")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for m in models_data:
        sm_steps, sm_mean, sm_std = _compute_smooth(m['steps'], m['stacked'], window)
        n_seeds = m['stacked'].shape[0]

        ax.fill_between(sm_steps, sm_mean - sm_std, sm_mean + sm_std,
                        alpha=0.18, color=m['color'])
        ax.plot(sm_steps, sm_mean, color=m['color'], linewidth=2.2,
                label=f"{m['name']}  (n={n_seeds}, final={sm_mean[-1]:,.0f})")
        ax.axhline(float(sm_mean[-1]), color=m['txt_color'],
                   linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_title(f'Training Curves — SAC / MAAC / C-MAAC  ({N_AGENTS} docks)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Episode Reward (per agent)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}' if x == 0 else f'{x/1000:.0f},000'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    seeds = [42 + i for i in range(args.n_seeds)]

    maac_models = [str(Path(d) / 'model_best.pt') for d in args.maac_dirs]
    sac_models  = [str(Path(d) / 'model_best.pt') for d in args.sac_dirs] if args.sac_dirs else []
    cmdp_models = [str(Path(d) / 'model_best.pt') for d in args.cmdp_dirs] if args.cmdp_dirs else []

    _patch_truck_generator()

    S1 = SCENARIOS[0]
    subtitle_s1 = f'normal_10, 500kW, 10 docks  |  n_seeds={args.n_seeds}'

    # ════════════════════════════════════════════════════════════════════
    # 0) Training Curves — SAC / MAAC / C-MAAC
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('  [0] Training Curves')
    print('='*60)

    d0 = out_dir / '0_training_curves'
    d0.mkdir(exist_ok=True)
    plot_training_curves(args.maac_dirs, args.sac_dirs, args.cmdp_dirs,
                          args.sac_n_agents, d0 / 'training_curves_all.png',
                          window=args.smooth)

    # ════════════════════════════════════════════════════════════════════
    # 1) Random / Heuristic / SAC / MAAC / C-MAAC  @ S1
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('  [1] Baseline 5-way comparison @ S1')
    print('='*60)

    d1 = out_dir / '1_baseline_s1'
    d1.mkdir(exist_ok=True)

    all_data_s1 = {}

    print("  Random...", flush=True)
    all_data_s1['Random'] = aggregate([eval_random(S1, s) for s in seeds])

    print("  Heuristic...", flush=True)
    all_data_s1['Heuristic'] = aggregate([eval_heuristic(S1, s) for s in seeds])

    if sac_models:
        print("  SAC...", flush=True)
        data = []
        for mp in sac_models:
            if Path(mp).exists():
                data += [eval_sac(mp, S1, s) for s in seeds]
        if data:
            all_data_s1['SAC'] = aggregate(data)

    print("  MAAC...", flush=True)
    data = []
    for mp in maac_models:
        if Path(mp).exists():
            data += [eval_maac(mp, S1, s) for s in seeds]
    if data:
        all_data_s1['MAAC'] = aggregate(data)

    if cmdp_models:
        print("  C-MAAC...", flush=True)
        data = []
        for mp in cmdp_models:
            if Path(mp).exists():
                data += [eval_cmdp(mp, S1, s) for s in seeds]
        if data:
            all_data_s1['C-MAAC'] = aggregate(data)

    models5 = [m for m in ['Random', 'Heuristic', 'SAC', 'MAAC', 'C-MAAC'] if m in all_data_s1]

    # Total Penalty (per agent)
    vals = [all_data_s1[m]['total_penalty'][0] / N_AGENTS for m in models5]
    stds = [all_data_s1[m]['total_penalty'][1] / N_AGENTS for m in models5]
    plot_single_metric_bar(models5, vals, stds,
                           'Total Penalty per Agent', 'Penalty (lower=better)',
                           d1 / '1_total_penalty.png', subtitle_s1)

    # 6 penalty components (per agent)
    for idx, (key, label) in enumerate(zip(PENALTY_KEYS, PENALTY_LABELS), start=2):
        vals = [all_data_s1[m][key][0] / N_AGENTS for m in models5]
        stds = [all_data_s1[m][key][1] / N_AGENTS for m in models5]
        plot_single_metric_bar(models5, vals, stds,
                               f'{label} per Agent', 'Penalty (lower=better)',
                               d1 / f'{idx}_{key}.png', subtitle_s1)

    # Overload Steps / Forced Departures - 두 그래프 y축 스케일 통일
    ol_vals = [all_data_s1[m]['overload_steps'][0] for m in models5]
    ol_stds = [all_data_s1[m]['overload_steps'][1] for m in models5]
    fd_vals = [all_data_s1[m]['forced_dep'][0] for m in models5]
    fd_stds = [all_data_s1[m]['forced_dep'][1] for m in models5]
    shared_top = max(
        max(v + s for v, s in zip(ol_vals, ol_stds)),
        max(v + s for v, s in zip(fd_vals, fd_stds)),
    ) * 1.18
    shared_top = max(shared_top, 1e-6)

    plot_single_metric_bar(models5, ol_vals, ol_stds,
                           'Grid Overload Steps', 'Steps (lower=better)',
                           d1 / '8_overload_steps.png', subtitle_s1, ylim_max=shared_top)

    plot_single_metric_bar(models5, fd_vals, fd_stds,
                           'Forced Departures', 'Count (lower=better)',
                           d1 / '9_forced_departures.png', subtitle_s1, ylim_max=shared_top)

    # ════════════════════════════════════════════════════════════════════
    # 2) SAC vs MAAC @ S1  /  3) SAC vs MAAC scenarios
    # ════════════════════════════════════════════════════════════════════
    if sac_models:
        print('\n' + '='*60)
        print('  [2,3] SAC vs MAAC')
        print('='*60)

        d2 = out_dir / '2_sac_vs_maac_s1'
        d3 = out_dir / '3_sac_vs_maac_scenarios'
        d2.mkdir(exist_ok=True); d3.mkdir(exist_ok=True)

        metric_defs = [
            ('total_penalty', 'Total Penalty per Agent', 'Penalty (lower=better)', 1.0 / N_AGENTS, '1_total_penalty.png'),
            ('overload_steps', 'Grid Overload Steps',     'Steps (lower=better)',  1.0,            '2_overload.png'),
            ('forced_dep',     'Forced Departures',       'Count (lower=better)',  1.0,            '3_forced_departures.png'),
        ]

        # @S1 (이미 계산된 all_data_s1 재사용)
        if 'SAC' in all_data_s1 and 'MAAC' in all_data_s1:
            # Total Penalty
            key, title, ylabel, scale, fname = metric_defs[0]
            vals = [all_data_s1['SAC'][key][0] * scale, all_data_s1['MAAC'][key][0] * scale]
            stds = [all_data_s1['SAC'][key][1] * scale, all_data_s1['MAAC'][key][1] * scale]
            plot_single_metric_bar(['SAC', 'MAAC'], vals, stds, title, ylabel,
                                   d2 / fname, subtitle_s1)

            # Overload Steps / Forced Departures - y축 스케일 통일
            ol_key, ol_title, ol_ylabel, _, ol_fname = metric_defs[1]
            fd_key, fd_title, fd_ylabel, _, fd_fname = metric_defs[2]
            ol_vals = [all_data_s1['SAC'][ol_key][0], all_data_s1['MAAC'][ol_key][0]]
            ol_stds = [all_data_s1['SAC'][ol_key][1], all_data_s1['MAAC'][ol_key][1]]
            fd_vals = [all_data_s1['SAC'][fd_key][0], all_data_s1['MAAC'][fd_key][0]]
            fd_stds = [all_data_s1['SAC'][fd_key][1], all_data_s1['MAAC'][fd_key][1]]
            shared_top = max(
                max(v + s for v, s in zip(ol_vals, ol_stds)),
                max(v + s for v, s in zip(fd_vals, fd_stds)),
            ) * 1.18
            shared_top = max(shared_top, 1e-6)

            plot_single_metric_bar(['SAC', 'MAAC'], ol_vals, ol_stds, ol_title, ol_ylabel,
                                   d2 / ol_fname, subtitle_s1, ylim_max=shared_top)
            plot_single_metric_bar(['SAC', 'MAAC'], fd_vals, fd_stds, fd_title, fd_ylabel,
                                   d2 / fd_fname, subtitle_s1, ylim_max=shared_top)

            # 도크별 충전 완료율 비교
            comp_sac  = all_data_s1['SAC']['completed_docks'][0]
            forc_sac  = all_data_s1['SAC']['forced_docks'][0]
            comp_maac = all_data_s1['MAAC']['completed_docks'][0]
            forc_maac = all_data_s1['MAAC']['forced_docks'][0]

            with np.errstate(divide='ignore', invalid='ignore'):
                rate_sac  = np.nan_to_num(comp_sac  / (comp_sac  + forc_sac)  * 100)
                rate_maac = np.nan_to_num(comp_maac / (comp_maac + forc_maac) * 100)

            dock_labels = [f'Dock {i}' for i in range(N_AGENTS)]
            plot_dock_completion_rate(dock_labels, rate_sac, rate_maac, 'SAC', 'MAAC',
                                      'Per-Dock Charging Completion Rate — SAC vs MAAC',
                                      d2 / '4_dock_completion.png', subtitle_s1)

        # 시나리오별
        sac_by_scen, maac_by_scen = {}, {}
        for idx, sc in enumerate(SCENARIOS, 1):
            name = sc['name']
            print(f"  [{idx}/{len(SCENARIOS)}] {name}", flush=True)

            if name == 'S1' and 'SAC' in all_data_s1 and 'MAAC' in all_data_s1:
                sac_by_scen[name]  = all_data_s1['SAC']
                maac_by_scen[name] = all_data_s1['MAAC']
                continue

            sac_data, maac_data = [], []
            for mp in sac_models:
                if Path(mp).exists():
                    sac_data += [eval_sac(mp, sc, s) for s in seeds]
            for mp in maac_models:
                if Path(mp).exists():
                    maac_data += [eval_maac(mp, sc, s) for s in seeds]
            if sac_data:
                sac_by_scen[name] = aggregate(sac_data)
            if maac_data:
                maac_by_scen[name] = aggregate(maac_data)

        names  = [sc['name']  for sc in SCENARIOS if sc['name'] in sac_by_scen and sc['name'] in maac_by_scen]
        labels = [sc['label'] for sc in SCENARIOS if sc['name'] in names]
        sub_scen = f'normal_10 base, 10 docks  |  n_seeds={args.n_seeds}'

        # Total Penalty
        key, title, ylabel, scale, fname = metric_defs[0]
        valsA = [sac_by_scen[n][key][0]  * scale for n in names]
        stdsA = [sac_by_scen[n][key][1]  * scale for n in names]
        valsB = [maac_by_scen[n][key][0] * scale for n in names]
        stdsB = [maac_by_scen[n][key][1] * scale for n in names]
        plot_scenario_bar(names, labels, valsA, stdsA, valsB, stdsB, 'SAC', 'MAAC',
                          title + ' by Scenario', ylabel, d3 / fname, sub_scen)

        # Overload Steps / Forced Departures - y축 스케일 통일
        ol_key, ol_title, ol_ylabel, _, ol_fname = metric_defs[1]
        fd_key, fd_title, fd_ylabel, _, fd_fname = metric_defs[2]
        ol_A = [sac_by_scen[n][ol_key][0]  for n in names]; ol_A_s = [sac_by_scen[n][ol_key][1]  for n in names]
        ol_B = [maac_by_scen[n][ol_key][0] for n in names]; ol_B_s = [maac_by_scen[n][ol_key][1] for n in names]
        fd_A = [sac_by_scen[n][fd_key][0]  for n in names]; fd_A_s = [sac_by_scen[n][fd_key][1]  for n in names]
        fd_B = [maac_by_scen[n][fd_key][0] for n in names]; fd_B_s = [maac_by_scen[n][fd_key][1] for n in names]
        shared_top = max(
            max(v + s for v, s in zip(ol_A, ol_A_s)), max(v + s for v, s in zip(ol_B, ol_B_s)),
            max(v + s for v, s in zip(fd_A, fd_A_s)), max(v + s for v, s in zip(fd_B, fd_B_s)),
        ) * 1.18
        shared_top = max(shared_top, 1e-6)

        plot_scenario_bar(names, labels, ol_A, ol_A_s, ol_B, ol_B_s, 'SAC', 'MAAC',
                          ol_title + ' by Scenario', ol_ylabel, d3 / ol_fname, sub_scen, ylim_max=shared_top)
        plot_scenario_bar(names, labels, fd_A, fd_A_s, fd_B, fd_B_s, 'SAC', 'MAAC',
                          fd_title + ' by Scenario', fd_ylabel, d3 / fd_fname, sub_scen, ylim_max=shared_top)

    # ════════════════════════════════════════════════════════════════════
    # 4) MAAC vs C-MAAC @ S1  /  5) MAAC vs C-MAAC scenarios
    # ════════════════════════════════════════════════════════════════════
    if cmdp_models:
        print('\n' + '='*60)
        print('  [4,5] MAAC vs C-MAAC')
        print('='*60)

        d4 = out_dir / '4_maac_vs_cmdp_s1'
        d5 = out_dir / '5_maac_vs_cmdp_scenarios'
        d4.mkdir(exist_ok=True); d5.mkdir(exist_ok=True)

        metric_defs = [
            ('total_penalty', 'Total Penalty per Agent', 'Penalty (lower=better)', 1.0 / N_AGENTS, '1_total_penalty.png'),
            ('overload_steps', 'Grid Overload Steps',     'Steps (lower=better)',  1.0,            '2_overload.png'),
            ('forced_dep',     'Forced Departures',       'Count (lower=better)',  1.0,            '3_forced_departures.png'),
        ]

        if 'MAAC' in all_data_s1 and 'C-MAAC' in all_data_s1:
            # Total Penalty
            key, title, ylabel, scale, fname = metric_defs[0]
            vals = [all_data_s1['MAAC'][key][0] * scale, all_data_s1['C-MAAC'][key][0] * scale]
            stds = [all_data_s1['MAAC'][key][1] * scale, all_data_s1['C-MAAC'][key][1] * scale]
            plot_single_metric_bar(['MAAC', 'C-MAAC'], vals, stds, title, ylabel,
                                   d4 / fname, subtitle_s1)

            # Overload Steps / Forced Departures - y축 스케일 통일
            ol_key, ol_title, ol_ylabel, _, ol_fname = metric_defs[1]
            fd_key, fd_title, fd_ylabel, _, fd_fname = metric_defs[2]
            ol_vals = [all_data_s1['MAAC'][ol_key][0], all_data_s1['C-MAAC'][ol_key][0]]
            ol_stds = [all_data_s1['MAAC'][ol_key][1], all_data_s1['C-MAAC'][ol_key][1]]
            fd_vals = [all_data_s1['MAAC'][fd_key][0], all_data_s1['C-MAAC'][fd_key][0]]
            fd_stds = [all_data_s1['MAAC'][fd_key][1], all_data_s1['C-MAAC'][fd_key][1]]
            shared_top = max(
                max(v + s for v, s in zip(ol_vals, ol_stds)),
                max(v + s for v, s in zip(fd_vals, fd_stds)),
            ) * 1.18
            shared_top = max(shared_top, 1e-6)

            plot_single_metric_bar(['MAAC', 'C-MAAC'], ol_vals, ol_stds, ol_title, ol_ylabel,
                                   d4 / ol_fname, subtitle_s1, ylim_max=shared_top)
            plot_single_metric_bar(['MAAC', 'C-MAAC'], fd_vals, fd_stds, fd_title, fd_ylabel,
                                   d4 / fd_fname, subtitle_s1, ylim_max=shared_top)

        maac_by_scen2, cmdp_by_scen = {}, {}
        for idx, sc in enumerate(SCENARIOS, 1):
            name = sc['name']
            print(f"  [{idx}/{len(SCENARIOS)}] {name}", flush=True)

            if name == 'S1' and 'MAAC' in all_data_s1 and 'C-MAAC' in all_data_s1:
                maac_by_scen2[name] = all_data_s1['MAAC']
                cmdp_by_scen[name]  = all_data_s1['C-MAAC']
                continue

            maac_data, cmdp_data = [], []
            for mp in maac_models:
                if Path(mp).exists():
                    maac_data += [eval_maac(mp, sc, s) for s in seeds]
            for mp in cmdp_models:
                if Path(mp).exists():
                    cmdp_data += [eval_cmdp(mp, sc, s) for s in seeds]
            if maac_data:
                maac_by_scen2[name] = aggregate(maac_data)
            if cmdp_data:
                cmdp_by_scen[name] = aggregate(cmdp_data)

        names  = [sc['name']  for sc in SCENARIOS if sc['name'] in maac_by_scen2 and sc['name'] in cmdp_by_scen]
        labels = [sc['label'] for sc in SCENARIOS if sc['name'] in names]
        sub_scen = f'normal_10 base, 10 docks  |  n_seeds={args.n_seeds}'

        # Total Penalty
        key, title, ylabel, scale, fname = metric_defs[0]
        valsA = [maac_by_scen2[n][key][0] * scale for n in names]
        stdsA = [maac_by_scen2[n][key][1] * scale for n in names]
        valsB = [cmdp_by_scen[n][key][0]  * scale for n in names]
        stdsB = [cmdp_by_scen[n][key][1]  * scale for n in names]
        plot_scenario_bar(names, labels, valsA, stdsA, valsB, stdsB, 'MAAC', 'C-MAAC',
                          title + ' by Scenario', ylabel, d5 / fname, sub_scen)

        # Overload Steps / Forced Departures - y축 스케일 통일
        ol_key, ol_title, ol_ylabel, _, ol_fname = metric_defs[1]
        fd_key, fd_title, fd_ylabel, _, fd_fname = metric_defs[2]
        ol_A = [maac_by_scen2[n][ol_key][0] for n in names]; ol_A_s = [maac_by_scen2[n][ol_key][1] for n in names]
        ol_B = [cmdp_by_scen[n][ol_key][0]  for n in names]; ol_B_s = [cmdp_by_scen[n][ol_key][1]  for n in names]
        fd_A = [maac_by_scen2[n][fd_key][0] for n in names]; fd_A_s = [maac_by_scen2[n][fd_key][1] for n in names]
        fd_B = [cmdp_by_scen[n][fd_key][0]  for n in names]; fd_B_s = [cmdp_by_scen[n][fd_key][1]  for n in names]
        shared_top = max(
            max(v + s for v, s in zip(ol_A, ol_A_s)), max(v + s for v, s in zip(ol_B, ol_B_s)),
            max(v + s for v, s in zip(fd_A, fd_A_s)), max(v + s for v, s in zip(fd_B, fd_B_s)),
        ) * 1.18
        shared_top = max(shared_top, 1e-6)

        plot_scenario_bar(names, labels, ol_A, ol_A_s, ol_B, ol_B_s, 'MAAC', 'C-MAAC',
                          ol_title + ' by Scenario', ol_ylabel, d5 / ol_fname, sub_scen, ylim_max=shared_top)
        plot_scenario_bar(names, labels, fd_A, fd_A_s, fd_B, fd_B_s, 'MAAC', 'C-MAAC',
                          fd_title + ' by Scenario', fd_ylabel, d5 / fd_fname, sub_scen, ylim_max=shared_top)

    print(f"\n  완료: {out_dir}/")


def _patch_truck_generator():
    import envs.ev_charging.ev_charging_env as env_mod
    import envs.ev_charging.truck_generator as tg
    env_mod.generate_arrival_schedule = tg.generate_arrival_schedule
    env_mod.get_arrival_rates         = tg.get_arrival_rates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maac_dirs', nargs='+', required=True)
    parser.add_argument('--sac_dirs',  nargs='+', default=None)
    parser.add_argument('--cmdp_dirs', nargs='+', default=None)
    parser.add_argument('--n_seeds',   type=int, default=50)
    parser.add_argument('--out_dir',   default='full_compare')
    parser.add_argument('--sac_n_agents', type=int, default=N_AGENTS,
                        help='SAC 보상 정규화용 에이전트 수 (기본 10)')
    parser.add_argument('--smooth',    type=int, default=300,
                        help='학습 곡선 스무딩 윈도우 (에피소드 수)')
    args = parser.parse_args()
    main(args)
