"""
EV 트럭 충전 스케줄링 멀티에이전트 환경

에이전트: N개 충전 도크 (각각 독립적으로 충전 전력 결정)
Action: 이산 충전 전력 {0, 10, 20, ..., 90} kW
에피소드: 24시간 = 144 타임스텝 (10분 간격)
"""
import numpy as np
from collections import deque
from gym.spaces import Box, Discrete

from envs.ev_charging.price_schedule import get_price_schedule, MAX_PRICE
from envs.ev_charging.truck_generator import (generate_arrival_schedule, get_arrival_rates)

# ============================================================
# 상수 정의
# ============================================================
EPISODE_LENGTH = 144          # 24시간 / 10분
DELTA_T = 1.0 / 6.0          # 10분 = 1/6시간

DEFAULT_NUM_DOCKS = 10        # 도크(에이전트) 수
DEFAULT_CHARGER_MAX_KW = 90.0 # 충전기의 최대 충전 전력 (kW)
DEFAULT_BATTERY_CAP = 60.0    # 기본 전기 트럭 배터리 용량 (kWh)
GRID_LIMIT_KW = 500.0         # 전력 상한 (kW, 10도크 기준: 10 × 50kW)

NUM_ACTIONS = 10              # {0, 10, 20, ..., 90} kW
ACTION_KW = [float(i * 10) for i in range(NUM_ACTIONS)]

# Reward 계수
KAPPA = -20000.0              # 부족 충전 페널티 (출발 시 1회)
P_OVERTIME = 15000.0          # 초과 체류 페널티 계수
P_WAIT = 15000.0              # 대기 페널티 계수
P_OVERLOAD = 285.6            # 과부하 페널티 계수 (KEPCO 최대부하 190.4원/kWh × 1.5, 제67조의3)

# 기대 SoC 곡선 파라미터
# K1: 최종 목표 충전 비율
# K2: 마감 지연에 대한 압박 곡선의 기울기
K1 = 1.0
K2 = 4.0

# 관측 벡터 차원
OBS_DIM = 16

# 정규화용 상수
MAX_QUEUE = 20                # 대기열 최대 크기 (정규화용)
MAX_STAY_STEPS = 15           # 최대 체류 스텝 (정규화용)
MAX_LOAD_STEPS = 10           # 최대 적재 스텝 (정규화용)


class EVChargingEnv:
    """
    EV 트럭 충전 스케줄링 멀티에이전트 환경

    MAAC 학습 코드와 호환되는 인터페이스:
      - observation_space: List[Box]
      - action_space: List[Discrete]
      - agents: List
      - reset() → obs
      - step(actions) → (obs, rewards, dones, infos)
      - seed(int)
    """

    def __init__(self, num_docks=DEFAULT_NUM_DOCKS, arrival_mode='normal'):
        self.num_docks = num_docks
        self.n = num_docks
        self.arrival_mode = arrival_mode

        # Gym spaces (에이전트마다 하나씩 생성)
        self.observation_space = [
            Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
            for _ in range(num_docks)
        ]
        self.action_space = [
            Discrete(NUM_ACTIONS) for _ in range(num_docks)
        ]

        # env_wrappers 호환용 에이전트 리스트
        self.agents = [
            type('Agent', (), {'adversary': False})()
            for _ in range(num_docks)
        ]

        # 랜덤 시드
        self.np_random = np.random.RandomState()

        # 도착률 (expected arrivals 계산용)
        self._arrival_rates = get_arrival_rates(mode=arrival_mode)

        # 내부 상태 (reset에서 초기화)
        self.current_step = 0
        self.dock_states = None
        self.arrival_queue = None
        self.price_schedule = None

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    # ============================================================
    # reset
    # ============================================================
    def reset(self):
        self.current_step = 0
        self.last_ep_penalties = getattr(self, 'ep_penalties', [])  # 이전 에피소드 보존
        self.ep_penalties = []  # 새 에피소드 페널티 초기화

        # 에피소드 스케줄 생성
        self.price_schedule = get_price_schedule()

        # 도크 초기화 (모두 빈자리)
        self.dock_states = [self._empty_dock() for _ in range(self.num_docks)]

        # 트럭 도착 스케줄 생성 및 대기열 구성
        trucks = generate_arrival_schedule(self.np_random, mode=self.arrival_mode)
        self.arrival_queue = deque(
            sorted(trucks, key=lambda x: x['arrival_step'])
        )

        # 스텝 0에 도착한 트럭 배정
        self._process_new_arrivals()

        return self._get_all_observations()

    # ============================================================
    # step
    # ============================================================
    def step(self, actions):
        actions = self._parse_actions(actions)

        # Phase 1: Action masking + 충전 적용 + 적재 갱신
        actual_powers = np.zeros(self.num_docks)
        for i in range(self.num_docks):
            action_kw = ACTION_KW[actions[i]]
            actual_powers[i] = self._apply_action_masking(i, action_kw)
            self._apply_charging(i, actual_powers[i])
            self._update_loading(i)

        # Phase 2: 출차 조건 판정 (reward 계산 전에 flag 설정)
        departure_flags = self._check_departures()

        # Phase 3: Reward 계산
        rewards, penalty_infos = self._compute_rewards(actual_powers, departure_flags)

        # Phase 4: 상태 전이 실행 (출차 도크 초기화)
        for i in range(self.num_docks):
            self._execute_transition(i, departure_flags[i])

        # Phase 5: 새 트럭 배정
        self._process_new_arrivals()

        # Phase 6: 시간 전진
        self.current_step += 1

        # 에피소드 종료 판정
        done = self.current_step >= EPISODE_LENGTH
        dones = np.array([done] * self.num_docks)
        obs = self._get_all_observations()
        infos = penalty_infos

        return obs, rewards, dones, infos

    # ============================================================
    # Action 처리
    # ============================================================
    def _parse_actions(self, actions):
        """one-hot 또는 정수 액션을 인덱스로 변환"""
        parsed = []
        for a in actions:
            if isinstance(a, np.ndarray):
                parsed.append(int(np.argmax(a)))
            else:
                parsed.append(int(a))
        return parsed

    def _apply_action_masking(self, i, action_kw):
        """Action Masking: 무효 액션을 0kW로 강제"""
        dock = self.dock_states[i]
        # 빈 도크 → 0kW
        if dock['connected'] == 0:
            return 0.0
        # 충전 완료 + 적재 중 → 0kW
        if dock['soc'] >= dock['target_soc'] and dock['loading_status'] == 1:
            return 0.0
        # 허용 범위 내로 클램프
        max_allowed = min(dock['charger_max_kw'], dock['ev_max_kw'])
        return min(action_kw, max_allowed)

    # ============================================================
    # 충전 및 적재 갱신
    # ============================================================
    def _apply_charging(self, i, power_kw):
        """SoC 갱신: B_{t+1} = min(1.0, B_t + (a × Δt) / E)"""
        dock = self.dock_states[i]
        if dock['connected'] == 0 or dock['battery_cap'] <= 0:
            return
        energy_kwh = power_kw * DELTA_T
        soc_increase = energy_kwh / dock['battery_cap']
        dock['soc'] = min(1.0, dock['soc'] + soc_increase)
        dock['current_power'] = power_kw

    def _update_loading(self, i):
        """적재 시간 감소 및 완료 판정"""
        dock = self.dock_states[i]
        if dock['connected'] == 0:
            return
        if dock['loading_status'] == 1 and dock['loading_remain'] > 0:
            dock['loading_remain'] -= 1 # 잔여 시간 1스텝 감소
            if dock['loading_remain'] <= 0:
                dock['loading_status'] = 0 # 적재 완료

    # ============================================================
    # 상태 전이
    # ============================================================
    def _check_departures(self):
        """
        각 도크의 출차/잔류 판정
        반환: list of str
          'none': 변화 없음 (Case 1: 정상진행, Case 4: 초과체류)
          'normal': Case 2 정상출차
          'forced': Case 3 강제출차 (충전 미완료)
        """
        flags = []
        for i in range(self.num_docks):
            dock = self.dock_states[i]
            if dock['connected'] == 0:
                flags.append('none')
                continue

            charge_done = dock['soc'] >= dock['target_soc']
            load_done = dock['loading_status'] == 0
            time_left = dock['departure_remain'] > 0

            # Case 2: 정상 출차 (충전+적재 모두 완료)
            if charge_done and load_done:
                flags.append('normal')
            # Case 3: 강제 출차 (시간 소진, 적재 완료, 충전 미완료)
            elif not time_left and load_done and not charge_done:
                flags.append('forced')
            # Case 1 or 4: 잔류
            else:
                dock['departure_remain'] -= 1
                flags.append('none')

        return flags

    def _execute_transition(self, i, flag):
        """출차 플래그에 따라 도크 초기화"""
        if flag in ('normal', 'forced'):
            self.dock_states[i] = self._empty_dock()

    # ============================================================
    # Reward 계산
    # ============================================================
    def _compute_rewards(self, actual_powers, departure_flags):
        rewards = np.zeros(self.num_docks)
        penalty_infos = []
        total_power = np.sum(actual_powers)
        step_idx = min(self.current_step, EPISODE_LENGTH - 1)
        price = self.price_schedule[step_idx]
        grid_limit = GRID_LIMIT_KW

        # 대기 트럭 수 (현재 시점 이전에 도착했지만 아직 대기 중인 트럭)
        queue_size = sum(
            1 for t in self.arrival_queue if t['arrival_step'] <= self.current_step
        )

        for i in range(self.num_docks):
            dock = self.dock_states[i]
            r = 0.0
            p1 = p2 = p3 = p4 = p5 = p6 = 0.0

            # 1. 충전 비용 페널티: -(p × a × Δt)
            p1 = price * actual_powers[i] * DELTA_T
            r -= p1

            # 2. 불만족 페널티 (매 스텝, 연결 시만)
            if dock['connected'] == 1:
                b_expected = self._expected_soc(i)
                if dock['soc'] < b_expected:
                    p2 = ((b_expected - dock['soc']) ** 2) * 8244
                    r -= p2

            # 3. 기대 충전량 미달 페널티 (강제 출차 시만)
            if departure_flags[i] == 'forced':
                p3 = -KAPPA  # KAPPA는 음수이므로 양수로 저장
                r += KAPPA

            # 4. 과부하 페널티
            if total_power > grid_limit:
                agent_share = actual_powers[i] / total_power
                p4 = agent_share * P_OVERLOAD * (total_power - grid_limit) * DELTA_T
                r -= p4

            # 5. 대기 페널티 (연결됐고 충전 여력 있는 도크만)
            # 빈 도크, 충전 완료 도크는 물리적으로 충전 불가 → 제외
            def _can_charge(j):
                d = self.dock_states[j]
                return (d['connected'] == 1 and d['soc'] < d['target_soc'])
            total_unused = sum(
                self.dock_states[j]['charger_max_kw'] - actual_powers[j]
                for j in range(self.num_docks)
                if _can_charge(j)
            )
            if queue_size > 0 and total_unused > 0 and _can_charge(i):
                agent_unused = dock['charger_max_kw'] - actual_powers[i]
                p5 = (agent_unused / total_unused) * P_WAIT * queue_size * DELTA_T
                r -= p5

            # 6. 초과 체류 페널티: -p^o × |t^r| × Δt (선형)
            if dock['departure_remain'] < 0 and dock['loading_status'] == 1:
                p6 = P_OVERTIME * abs(dock['departure_remain']) * DELTA_T
                r -= p6

            rewards[i] = r
            penalty_infos.append({
                'charging_cost': p1,
                'dissatisfaction': p2,
                'undercharge': p3,
                'overload': p4,
                'waiting': p5,
                'overtime': p6,
                'actual_power': actual_powers[i],  # 출차 후 리셋 전 실제 충전 전력
            })

        self.ep_penalties.append(penalty_infos)
        return rewards, penalty_infos

    # 기대 SoC 곡선: 불만족 페널티에서 사용
    def _expected_soc(self, i):
        """기대 SoC 곡선 B^a (지수 함수)"""
        dock = self.dock_states[i]
        if dock['connected'] == 0:
            return 0.0
        t_a = dock['dock_arrival_time'] # 도착 시간
        t_d = dock['departure_deadline'] # 출발 마감 시간
        if t_d <= t_a:
            return dock['target_soc']
        progress = (self.current_step - t_a) / (t_d - t_a)
        progress = max(0.0, min(1.0, progress)) # 체류 시간 몇 % 지났는지 확인
        denominator = np.exp(-K2) - 1.0  # K2>0: 초반 충전 압박 강화 (지수 감소 곡선, 단거리 물류 트럭에 적합)

        # K2가 0에 가까워서 분모가 0이 될 경우 나누기 에러 방지. 이때는 선형으로 대체
        if abs(denominator) < 1e-10:  
            return dock['target_soc'] * progress
        b_a = dock['target_soc'] * K1 * (np.exp(-K2 * progress) - 1.0) / denominator
        return max(0.0, b_a)

    # ============================================================
    # 대기열 관리
    # ============================================================
    def _process_new_arrivals(self):
        """대기열에서 빈 도크에 트럭 배정 (FIFO)"""
        for i in range(self.num_docks):
            if self.dock_states[i]['connected'] == 0:
                if self.arrival_queue:
                    truck = self.arrival_queue[0]
                    if truck['arrival_step'] <= self.current_step:
                        self.arrival_queue.popleft()
                        self._assign_truck(i, truck)

    def _assign_truck(self, dock_i, truck):
        """트럭을 도크에 배정 (대기 시간만큼 departure_remain 감소)"""
        self.dock_states[dock_i] = {
            'connected': 1,
            'soc': truck['initial_soc'],
            'target_soc': truck['target_soc'],
            'departure_deadline': truck['departure_deadline'],
            'departure_remain': truck['departure_deadline'] - self.current_step,
            'dock_arrival_time': self.current_step,
            'loading_remain': truck['loading_time'],
            'loading_status': 1,
            'battery_cap': truck['battery_cap'],
            'charger_max_kw': DEFAULT_CHARGER_MAX_KW,
            'ev_max_kw': truck['ev_max_kw'],
            'current_power': 0.0,
        }

    def _empty_dock(self):
        """빈 도크 상태"""
        return {
            'connected': 0,
            'soc': 0.0,
            'target_soc': 0.0,
            'departure_deadline': 0,
            'departure_remain': 0,
            'dock_arrival_time': 0,
            'loading_remain': 0,
            'loading_status': 0,
            'battery_cap': 0.0,
            'charger_max_kw': DEFAULT_CHARGER_MAX_KW,
            'ev_max_kw': 0.0,
            'current_power': 0.0,
        }

    # ============================================================
    # 관측 벡터 구성
    # ============================================================
    def _get_all_observations(self):
        return np.array([self._get_observation(i) for i in range(self.num_docks)],
                        dtype=np.float32)

    def _get_observation(self, i):
        """
        에이전트 i의 관측 벡터 (16차원)
        [0] 시간, [1] 전력상한, [2] 대기열, [3] 예상도착, [4] 요금,
        [5] 연결여부, [6] SoC, [7] 목표SoC, [8] 출발마감 남은시간,
        [9] 도크 배정시간, [10] 적재 남은시간, [11] 적재상태,
        [12] 배터리용량, [13] 충전기최대, [14] EV최대, [15] 현재전력
        """
        dock = self.dock_states[i]
        step_idx = min(self.current_step, EPISODE_LENGTH - 1)

        # 대기 트럭 수
        queue_size = sum(
            1 for t in self.arrival_queue if t['arrival_step'] <= self.current_step
        )
        # 다음 스텝 예상 도착 수
        next_step = self.current_step + 1
        if next_step < EPISODE_LENGTH:
            expected = self._arrival_rates[next_step]
        else:
            expected = 0.0


        # 0~1 사이로 정규화 (신경망 입력용)
        obs = np.array([
            self.current_step / (EPISODE_LENGTH - 1),                       # [0] 시간
            GRID_LIMIT_KW / GRID_LIMIT_KW,                                  # [1] 전력 상한
            min(queue_size, MAX_QUEUE) / MAX_QUEUE,                         # [2] 대기열
            min(expected, self.num_docks) / max(self.num_docks, 1),         # [3] 예상 도착 트럭 수
            self.price_schedule[step_idx] / MAX_PRICE,                      # [4] 요금
            float(dock['connected']),                                       # [5] 도크에 트럭이 있는지 없는지 여부
            dock['soc'],                                                    # [6] SoC
            dock['target_soc'],                                             # [7] 목표 SoC
            np.clip(dock['departure_remain'] / MAX_STAY_STEPS, -1.0, 1.0), # [8] 출발마감 남은시간
            dock['dock_arrival_time'] / (EPISODE_LENGTH - 1) if EPISODE_LENGTH > 1 else 0, # [9] 도크 배정시간
            min(dock['loading_remain'], MAX_LOAD_STEPS) / MAX_LOAD_STEPS,   # [10] 적재 남은시간
            float(dock['loading_status']),                                  # [11] 적재 상태
            dock['battery_cap'] / 100.0,                                    # [12] 배터리 용량
            dock['charger_max_kw'] / DEFAULT_CHARGER_MAX_KW,               # [13] 충전기 최대
            dock['ev_max_kw'] / DEFAULT_CHARGER_MAX_KW,                    # [14] EV 최대
            dock['current_power'] / DEFAULT_CHARGER_MAX_KW,                # [15] 현재 전력
        ], dtype=np.float32)

        return obs
