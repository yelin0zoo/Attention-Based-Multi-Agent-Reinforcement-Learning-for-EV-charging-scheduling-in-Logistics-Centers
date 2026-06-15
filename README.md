# 어텐션 기반 다중 에이전트 강화학습을 활용한 물류센터 EV 충전 전력 스케줄링

물류센터 내 EV 트럭 충전 도크(기본 10기)의 전력 스케줄링을 다중 에이전트 강화학습으로 학습하는 코드입니다.
[MAAC (Multi-Actor-Attention-Critic)](https://arxiv.org/abs/1810.02912)을 기반으로,
계약 전력(Grid Limit) 제약을 명시적으로 다루는 **C-MAAC (Constrained MAAC, CMDP + Lagrangian relaxation)**
를 제안하고, Random / Heuristic / SAC / MAAC 베이스라인과 비교합니다.

## 디렉토리 구조

```
EV-Charging-MAAC/
├── envs/ev_charging/
│   ├── ev_charging_env.py    # EV 충전 환경 (10도크, 24시간=144 스텝)
│   ├── price_schedule.py      # 한전 산업용(고압) 여름철 시간대별 요금
│   └── truck_generator.py     # 트럭 도착/적재량
│
├── algorithms/
│   ├── attention_sac.py        # MAAC
│   └── attention_sac_cmdp.py   # C-MAAC (제안 방법, CMDP + 라그랑주 완화)
│
├── utils/                       # 버퍼, 정책망, 환경 래퍼 등 공통 유틸
│   ├── agents.py
│   ├── buffer.py
│   ├── buffer_cmdp.py
│   ├── critics.py
│   ├── env_wrappers.py
│   ├── make_env.py
│   ├── misc.py
│   ├── policies.py
│   └── single_wrapper.py
│
├── train_maac.py                # MAAC 학습
├── train_cmdp.py                # C-MAAC 학습
├── train_sac.py                 # Single-Agent Discrete SAC 베이스라인 학습
│
├── eval_full_compare.py         # 종합 비교/시각화
├── benchmark_random.py          # Random 정책 베이스라인
└── benchmark_Heuristic.py        # 규칙 기반 Heuristic 베이스라인
```

## 환경 개요

- 도크 수: 10개 (`NUM_DOCKS`), 1 에피소드 = 144 스텝 (10분 간격 × 24시간)
- 행동: 도크별 충전 전력 0~90kW (10단계, `NUM_ACTIONS=10`)
- 보상: 충전비용(`charging_cost`), 불만족(`dissatisfaction`), 미충족충전(`undercharge`),
  과부하(`overload`), 대기(`waiting`), 출차지연(`overtime`) 6개 페널티 항목의 합
- 전기요금: 한전 산업용(을) 고압 여름철 요금 적용 (`envs/ev_charging/price_schedule.py`)
  - 경부하 (22:00 ~ 08:00): 79.2원/kWh
  - 중간부하 (08:00 ~ 11:00, 12:00 ~ 13:00, 18:00 ~ 22:00): 137.4원/kWh
  - 최대부하 (11:00 ~ 12:00, 13:00 ~ 18:00): 190.4원/kWh

## 학습

### MAAC
```shell
python train_maac.py --model_name maac_v2_s1 --seed 1 --n_episodes 20000
```

### C-MAAC (제안 방법)
```shell
python train_cmdp.py --model_name cmdp_v9_s1 --seed 1 \
    --arrival_mode normal_10 --n_episodes 20000 \
    --d1 0.14 --lambda_lr 0.005 --lambda_max 20 --num_updates 4
```

### SAC (베이스라인)
```shell
python train_sac.py --model_name sac_v1_s1 --seed 1 --n_episodes 20000
```

학습 결과는 `models/ev_charging/{model_name}/run1/` 아래에 체크포인트(`model_best.pt`, `incremental/`)와
TensorBoard 로그(`logs/`)로 저장됩니다.

## 평가 / 시각화

`eval_full_compare.py`는 학습된 모델들의 모든 비교 그래프를 생성합니다.

```shell
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
```

생성되는 그래프:

| 출력 폴더 | 내용 |
|---|---|
| `0_training_curves/` | SAC / MAAC / C-MAAC 학습 곡선 |
| `1_baseline_s1/` | Random/Heuristic/SAC/MAAC/C-MAAC, S1 기준 — Total Penalty, 6개 페널티 항목, Overload Steps, Forced Departures |
| `2_sac_vs_maac_s1/` | SAC vs MAAC, S1 — Total Penalty, Overload, Forced Departures + 도크 완료율 |
| `3_sac_vs_maac_scen/` | SAC vs MAAC, 시나리오별(S1~S10) |
| `4_maac_vs_cmdp_s1/` | MAAC vs C-MAAC, S1 |
| `5_maac_vs_cmdp_scen/` | MAAC vs C-MAAC, 시나리오별(S1~S10) |

## 참고

본 코드는 [Iqbal & Sha, "Actor-Attention-Critic for Multi-Agent Reinforcement Learning" (ICML 2019)](https://arxiv.org/abs/1810.02912)의
[공식 구현](https://github.com/shariqiqbal2810/MAAC)을 기반으로 작성되었습니다.
