"""
리플레이 버퍼 모듈
멀티에이전트 강화학습에서 경험 데이터(s, a, r, s', done)를 저장하고 샘플링하는 버퍼
병렬 환경에서 수집한 데이터를 효율적으로 관리함

특징:
- 순환 버퍼 방식: 가득 차면 가장 오래된 데이터부터 덮어씀
- 에이전트별 독립 저장: 각 에이전트의 관측/행동/보상을 별도 배열에 저장
- 보상 정규화: 샘플링 시 보상의 평균/표준편차로 정규화 가능
"""
import numpy as np
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer(object):
    """
    병렬 롤아웃을 지원하는 멀티에이전트 리플레이 버퍼

    Off-policy 알고리즘(SAC 등)에서 과거 경험을 저장하고
    랜덤 샘플링하여 학습에 사용하는 핵심 구성 요소
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        버퍼 초기화: 지정된 크기만큼 NumPy 배열을 미리 할당

        인자:
            max_steps (int): 버퍼에 저장할 수 있는 최대 타임스텝 수
            num_agents (int): 환경의 에이전트 수
            obs_dims (list of int): 각 에이전트의 관측 차원 리스트
            ac_dims (list of int): 각 에이전트의 행동 차원 리스트
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        # 각 에이전트별 저장 배열 초기화
        self.obs_buffs = []  # 현재 관측값 버퍼
        self.ac_buffs = []  # 행동 버퍼
        self.rew_buffs = []  # 보상 버퍼
        self.next_obs_buffs = []  # 다음 관측값 버퍼
        self.done_buffs = []  # 에피소드 종료 여부 버퍼
        for odim, adim in zip(obs_dims, ac_dims):
            # 각 에이전트마다 고정 크기의 NumPy 배열 할당 (메모리 효율적)
            self.obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))

        self.filled_i = 0  # 버퍼에 실제 데이터가 채워진 양 (가득 차면 max_steps)
        self.curr_i = 0  # 다음에 쓸 위치 인덱스 (순환하며 오래된 데이터를 덮어씀)

    def __len__(self):
        """버퍼에 저장된 유효한 데이터 수 반환"""
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        """
        새로운 경험 데이터를 버퍼에 저장
        병렬 환경에서 수집한 여러 타임스텝의 데이터를 한 번에 추가

        버퍼가 가득 찬 경우 np.roll로 데이터를 밀어내고 앞부터 다시 씀

        인자:
            observations (np.array): 관측값 [환경 수, 에이전트 수, 관측 차원]
            actions (list): 에이전트별 행동 리스트 [에이전트별 [환경 수, 행동 차원]]
            rewards (np.array): 보상 [환경 수, 에이전트 수]
            next_observations (np.array): 다음 관측값 [환경 수, 에이전트 수, 관측 차원]
            dones (np.array): 종료 여부 [환경 수, 에이전트 수]
        """
        nentries = observations.shape[0]  # 추가할 데이터 수 (병렬 환경 수)
        if self.curr_i + nentries > self.max_steps:
            # 버퍼 끝을 넘어가는 경우: 데이터를 앞으로 롤오버
            rollover = self.max_steps - self.curr_i  # 롤오버할 인덱스 수
            for agent_i in range(self.num_agents):
                # np.roll로 배열을 순환 이동시켜 빈 공간을 앞에 만듦
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0  # 쓰기 위치를 처음으로 리셋
            self.filled_i = self.max_steps  # 버퍼가 가득 찬 상태
        # 각 에이전트별로 데이터 저장
        for agent_i in range(self.num_agents):
            # 관측값: 환경별로 쌓아서 저장
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # 행동: 이미 에이전트별로 배치되어 있으므로 직접 저장
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            # 보상: 환경 × 에이전트 배열에서 해당 에이전트 열 추출
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            # 다음 관측값
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            # 종료 여부
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries  # 쓰기 위치 전진
        if self.filled_i < self.max_steps:
            self.filled_i += nentries  # 채워진 양 업데이트
        if self.curr_i == self.max_steps:
            self.curr_i = 0  # 버퍼 끝에 도달하면 처음으로 순환

    def sample(self, N, to_gpu=False, norm_rews=True):
        """
        버퍼에서 미니배치를 랜덤 샘플링

        인자:
            N (int): 샘플링할 데이터 수 (미니배치 크기)
            to_gpu (bool): True면 GPU 텐서로 변환 (기본값: False)
            norm_rews (bool): True면 보상을 평균/표준편차로 정규화 (기본값: True)
                정규화를 통해 학습 안정성을 높이고 보상 스케일에 덜 민감하게 만듦
        반환:
            tuple: (관측값 리스트, 행동 리스트, 보상 리스트, 다음 관측값 리스트, 종료 리스트)
                각 리스트는 에이전트별 PyTorch Variable을 포함
        """
        # 유효한 범위 내에서 N개의 인덱스를 랜덤 추출 (중복 허용)
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        # GPU/CPU에 따른 텐서 변환 함수 정의
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        # 보상 정규화 여부에 따라 처리
        if norm_rews:
            # (보상 - 평균) / 표준편차로 정규화
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             (self.rew_buffs[i][:self.filled_i].std() + 1e-6))
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        # 에이전트별로 샘플링된 데이터를 텐서로 변환하여 반환
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],       # 관측값
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],        # 행동
                ret_rews,                                                                # 보상
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],  # 다음 관측값
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])      # 종료 여부

    def get_average_rewards(self, N):
        """
        최근 N개 타임스텝의 평균 보상을 에이전트별로 계산
        에피소드 보상 로깅에 사용됨

        인자:
            N (int): 평균을 계산할 최근 타임스텝 수
                일반적으로 에피소드 길이 × 병렬 환경 수
        반환:
            list: 각 에이전트의 평균 보상 리스트
        """
        if self.filled_i == self.max_steps:
            # 버퍼가 가득 찬 경우: 음수 인덱싱 허용 (순환 참조)
            inds = np.arange(self.curr_i - N, self.curr_i)
        else:
            # 버퍼가 아직 안 찬 경우: 0 이하로 내려가지 않도록 보정
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
