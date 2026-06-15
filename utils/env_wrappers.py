"""
환경 래퍼 모듈
OpenAI Baselines의 벡터화 환경 코드를 멀티에이전트 환경에 맞게 수정한 것
여러 환경 인스턴스를 병렬 또는 순차적으로 실행하여 데이터 수집 효율을 높임

SubprocVecEnv: 각 환경을 별도 서브프로세스에서 병렬 실행 (빠르지만 오버헤드 있음)
DummyVecEnv: 단일 프로세스에서 순차 실행 (디버깅에 적합)
"""
import numpy as np
import cloudpickle  # 함수 직렬화를 위한 모듈
from multiprocessing import Process, Pipe  # 멀티프로세싱을 위한 프로세스와 파이프


# === baselines 의존성 제거: VecEnv, CloudpickleWrapper 직접 정의 ===
# OpenAI baselines가 더 이상 유지보수되지 않아 설치 불가능하므로,
# 실제로 사용하는 두 클래스만 직접 구현함

class CloudpickleWrapper(object):
    """
    cloudpickle을 사용하여 함수를 직렬화하는 래퍼
    서브프로세스 간에 환경 생성 함수를 전달할 때 사용
    (일반 pickle은 로컬 함수/람다를 직렬화하지 못함)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv(object):
    """
    벡터화 환경 기본 클래스 (baselines.common.vec_env.VecEnv 대체)
    여러 환경을 동시에 관리하기 위한 인터페이스 정의
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        """비동기 스텝 실행 후 결과 수집"""
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        raise NotImplementedError

    def step_wait(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        pass


def worker(remote, parent_remote, env_fn_wrapper):
    """
    서브프로세스에서 실행되는 환경 워커 함수
    메인 프로세스와 파이프(Pipe)를 통해 명령을 주고받으며 환경을 제어함

    지원하는 명령:
        'step': 환경에서 한 스텝 실행 (모든 에이전트가 done이면 자동 리셋)
        'reset': 환경 초기화
        'reset_task': 태스크 리셋
        'close': 환경 종료
        'get_spaces': 관측/행동 공간 정보 반환
        'get_agent_types': 각 에이전트의 타입(적대적/협력적) 반환

    인자:
        remote: 워커 측 파이프 (메인 프로세스와 통신)
        parent_remote: 부모 측 파이프 (워커에서는 닫아야 함)
        env_fn_wrapper: 환경 생성 함수를 감싼 CloudpickleWrapper
    """
    parent_remote.close()  # 워커에서는 부모 측 파이프를 닫음 (리소스 누수 방지)
    env = env_fn_wrapper.x()  # 래핑된 함수를 호출하여 환경 인스턴스 생성
    while True:
        cmd, data = remote.recv()  # 메인 프로세스로부터 명령 수신
        if cmd == 'step':
            # 환경에서 한 스텝 실행
            ob, reward, done, info = env.step(data)
            if all(done):
                # 모든 에이전트가 종료되면 환경을 자동으로 리셋
                ob = env.reset()
            remote.send((ob, reward, done, info))  # 결과를 메인 프로세스로 전송
        elif cmd == 'reset':
            ob = env.reset()  # 환경 초기화
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()  # 태스크 리셋 (환경 구성 변경)
            remote.send(ob)
        elif cmd == 'close':
            remote.close()  # 파이프 닫고 루프 종료
            break
        elif cmd == 'get_spaces':
            # 관측 공간과 행동 공간 정보를 반환
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            # 각 에이전트가 적대적(adversary)인지 일반 에이전트인지 판별
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                # adversary 속성이 없으면 모두 일반 에이전트로 분류
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError  # 지원하지 않는 명령은 에러 발생


class SubprocVecEnv(VecEnv):
    """
    서브프로세스 기반 벡터화 환경
    각 환경을 별도 프로세스에서 실행하여 진정한 병렬 처리를 구현함
    CPU 코어를 활용하여 데이터 수집 속도를 크게 향상시킴

    통신 방식: 파이프(Pipe)를 통한 프로세스 간 통신 (IPC)
    """
    def __init__(self, env_fns, spaces=None):
        """
        인자:
            env_fns (list): 각 환경을 생성하는 함수들의 리스트
            spaces: (미사용) 관측/행동 공간 오버라이드
        """
        self.waiting = False  # step_async 후 결과 대기 중인지 여부
        self.closed = False  # 환경이 종료되었는지 여부
        nenvs = len(env_fns)  # 환경 개수
        # 각 환경마다 양방향 파이프 생성 (메인↔워커)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        # 각 워커 프로세스 생성 (환경 함수를 CloudpickleWrapper로 직렬화)
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # 데몬 프로세스: 메인 프로세스 종료 시 자동 정리
            p.start()
        for remote in self.work_remotes:
            remote.close()  # 메인 프로세스에서는 워커 측 파이프를 닫음

        # 첫 번째 환경에서 공간 정보와 에이전트 타입 조회
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()  # 에이전트 타입 리스트 (예: ['agent', 'adversary'])
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        """
        비동기 스텝: 모든 환경에 동시에 행동 명령을 전송
        결과는 step_wait()에서 수집

        인자:
            actions (list): 각 환경별 행동 리스트
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True  # 결과 대기 상태로 전환

    def step_wait(self):
        """
        비동기 스텝의 결과를 수집
        모든 서브프로세스로부터 결과가 돌아올 때까지 대기

        반환:
            obs (np.array): 관측값 배열 [환경 수, 에이전트 수, 관측 차원]
            rews (np.array): 보상 배열 [환경 수, 에이전트 수]
            dones (np.array): 종료 여부 배열 [환경 수, 에이전트 수]
            infos (tuple): 추가 정보 튜플
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False  # 대기 상태 해제
        obs, rews, dones, infos = zip(*results)  # 결과를 항목별로 분리
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """
        모든 환경을 초기화하고 초기 관측값 반환

        반환:
            np.array: 모든 환경의 초기 관측값 [환경 수, 에이전트 수, 관측 차원]
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        """
        모든 환경의 태스크를 리셋

        반환:
            np.array: 리셋 후 관측값
        """
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        """
        모든 서브프로세스를 안전하게 종료
        대기 중인 결과가 있으면 먼저 수신한 후 종료 명령 전송
        """
        if self.closed:
            return
        if self.waiting:
            # 아직 수신하지 않은 결과가 있으면 먼저 받아서 정리
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))  # 각 워커에 종료 명령 전송
        for p in self.ps:
            p.join()  # 모든 워커 프로세스가 종료될 때까지 대기
        self.closed = True


class DummyVecEnv(VecEnv):
    """
    더미 벡터화 환경
    단일 프로세스에서 순차적으로 환경을 실행하는 간단한 래퍼
    병렬 처리 오버헤드가 없어 디버깅이나 단일 환경 실행에 적합
    SubprocVecEnv와 동일한 인터페이스를 제공함
    """
    def __init__(self, env_fns):
        """
        인자:
            env_fns (list): 환경 생성 함수 리스트
        """
        self.envs = [fn() for fn in env_fns]  # 모든 환경 즉시 생성
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        # 에이전트 타입 판별 (적대적/협력적)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')  # 각 환경의 현재 타임스텝 추적
        self.actions = None  # step_async에서 저장된 행동

    def step_async(self, actions):
        """
        비동기 스텝 (실제로는 동기): 행동을 저장해두고 step_wait에서 실행

        인자:
            actions (list): 각 환경별 행동 리스트
        """
        self.actions = actions

    def step_wait(self):
        """
        저장된 행동으로 모든 환경을 순차 실행
        에피소드가 끝난 환경은 자동으로 리셋됨

        반환:
            obs, rews, dones, infos: 관측, 보상, 종료여부, 추가정보
        """
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1  # 타임스텝 카운터 증가
        for (i, done) in enumerate(dones):
            if all(done):  # 모든 에이전트가 종료된 환경은 리셋
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0  # 타임스텝 카운터 초기화
        self.actions = None  # 행동 버퍼 초기화
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        """
        모든 환경 초기화

        반환:
            np.array: 초기 관측값 배열
        """
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        """환경 종료 (DummyVecEnv는 별도 정리 불필요)"""
        return
