import numpy as np

# 배터리 용량 (kWh) — 시나리오별로 패치 가능
BATTERY_CAP_KWH = 60.0

# 모드별 시간대별 도착률 λ (포아송 평균, 10도크 기준)
# 각 튜플: (minutes_end, normal_10, extreme, smooth, low)
_RATE_TABLE = [
    (  60, 0.83, 1.49, 0.60, 0.40),  # 00:00~01:00  야간조 2회전 적재
    ( 180, 0.33, 0.10, 0.40, 0.17),  # 01:00~03:00  배송 시작/플렉스 보충
    ( 240, 0.67, 1.17, 0.50, 0.34),  # 03:00~04:00  야간조 3회전 적재
    ( 420, 0.17, 0.06, 0.34, 0.06),  # 04:00~07:00  심야 마감 배송
    ( 540, 0.33, 0.10, 0.40, 0.17),  # 07:00~09:00  아침 교대
    ( 630, 1.00, 1.67, 0.67, 0.50),  # 09:00~10:30  주간조 1회전 (최대 피크)
    ( 870, 0.33, 0.10, 0.40, 0.17),  # 10:30~14:30  간헐적 적재
    ( 960, 0.83, 1.49, 0.60, 0.40),  # 14:30~16:00  주간조 2회전
    (1200, 0.33, 0.10, 0.40, 0.17),  # 16:00~20:00  배송 마무리
    (1260, 0.33, 0.10, 0.40, 0.17),  # 20:00~21:00  저녁 교대
    (1350, 1.00, 1.67, 0.67, 0.50),  # 21:00~22:30  야간조 1회전 (야간 최대 피크)
    (1440, 0.17, 0.06, 0.34, 0.06),  # 22:30~24:00  야간 배송 시작
]

_MODE_INDEX = {'normal_10': 1, 'extreme': 2, 'smooth': 3, 'low': 4}


def get_arrival_rates(mode='normal'):
    """
    24시간(144 타임스텝) 시간대별 트럭 도착률 (포아송 λ)

    모드 (10도크 기준):
      normal_10 - 일 ~65대, 최대 λ=1.00
      extreme   - 피크 시간대 집중 (일 ~69대, 최대 λ=1.67)
      smooth    - 시간대별 차이 완만 (일 ~64대, 최대 λ=0.67)
      low       - 전체적으로 λ 낮춤 (일 ~32대, 최대 λ=0.50)
    """
    if mode not in _MODE_INDEX:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_MODE_INDEX.keys())}")

    col = _MODE_INDEX[mode]
    rates = np.zeros(144)
    for t in range(144):
        minutes = t * 10
        for end_min, *values in _RATE_TABLE:
            if minutes < end_min:
                rates[t] = values[col - 1]
                break
    return rates


def generate_arrival_schedule(np_random, episode_length=144, mode='normal'):
    """
    24시간 에피소드의 트럭 도착 스케줄 생성

    인자:
      np_random: numpy RandomState
      episode_length: 에피소드 길이 (기본 144)
      mode: 도착률 모드 ('normal', 'extreme', 'smooth', 'low', 'normal_10', 'extreme_10')

    반환: list of dict, 각 트럭 정보:
      - arrival_step: 차고지 도착 타임스텝 (0~143)
      - departure_deadline: 출발 마감 타임스텝 (절대값)
      - initial_soc: 초기 SoC (0~1)
      - target_soc: 목표 SoC (0~1)
      - loading_time: 적재에 필요한 시간 (타임스텝)
      - num_items: 적재 물량 수
      - battery_cap: 배터리 용량 (kWh)
      - ev_max_kw: EV 최대 충전 전력 (kW)
    """
    rates = get_arrival_rates(mode)
    trucks = []

    for t in range(episode_length):
        n_arrivals = np_random.poisson(rates[t])
        for _ in range(n_arrivals):
            # 적재 물량 100~150건, 물건 1개당 30초 적재 시간 소요
            num_items = np_random.randint(100, 151)
            loading_time = int(np.ceil(num_items * 30 / 600))  # 타임스텝 변환 (600초=10분=1스텝)

            # 출발까지 남은 시간 = 적재 시간 + 여유 2~4스텝
            margin = np_random.randint(2, 5)
            total_time = loading_time + margin

            trucks.append({
                'arrival_step': t,
                'departure_deadline': t + total_time,
                'initial_soc': np_random.uniform(0.1, 0.4),
                'target_soc': np_random.uniform(0.6, 0.9),
                'loading_time': loading_time,
                'num_items': num_items,
                'battery_cap': BATTERY_CAP_KWH,
                'ev_max_kw': np_random.choice([60.0, 70.0, 80.0, 90.0]),
            })

    return trucks
