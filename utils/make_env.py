"""
멀티에이전트 환경 생성 모듈
시나리오 이름을 받아 해당하는 MultiAgentEnv 환경 객체를 생성함

사용 예시:
    env = make_env('simple_speaker_listener')

생성된 환경은 OpenAI Gym과 유사하게 사용 가능:
    obs = env.reset()
    next_obs, rewards, dones, infos = env.step(actions)

정책은 모든 에이전트의 행동을 리스트로 출력해야 함
각 행동은 (물리적 행동 차원 + 통신 차원) 크기의 NumPy 배열
"""

def make_env(scenario_name, benchmark=False, discrete_action=False, arrival_mode='normal'):
    """
    MultiAgentEnv 환경 객체를 생성하는 함수
    시나리오 스크립트를 로드하여 월드를 만들고, 환경을 초기화함

    인자:
        scenario_name (str): 시나리오 이름 (.py 확장자 제외)
            예: 'fullobs_collect_treasure', 'multi_speaker_listener', 'ev_charging'
        benchmark (bool): 벤치마킹 데이터 수집 여부 (보상, 관측값) (기본값: False)
            True면 평가 시 (보상, 관측값) + 추가 데이터를 기록
        discrete_action (bool): 이산 행동 공간 사용 여부 (기본값: False)

    유용한 환경 속성:
        .observation_space: 각 에이전트의 관측 공간
        .action_space: 각 에이전트의 행동 공간
        .n: 에이전트 수

    반환:
        MultiAgentEnv: 초기화된 멀티에이전트 환경 객체
    """
    # EV 충전 환경은 독립 Gym 환경 (multiagent.core 불필요)
    if scenario_name == 'ev_charging':
        from envs.ev_charging import EVChargingEnv
        return EVChargingEnv(arrival_mode=arrival_mode)

    from multiagent.environment import MultiAgentEnv  # 멀티에이전트 환경 클래스
    import multiagent.scenarios as old_scenarios  # 기본 시나리오 모듈
    import envs.mpe_scenarios as new_scenarios  # 커스텀 시나리오 모듈

    # 시나리오 스크립트 로드: 먼저 기본 시나리오에서 찾고, 없으면 커스텀 시나리오에서 로드
    try:
        scenario = old_scenarios.load(scenario_name + ".py").Scenario()
    except:
        scenario = new_scenarios.load(scenario_name + ".py").Scenario()

    # 시나리오의 make_world()를 호출하여 월드(에이전트, 랜드마크 등) 생성
    world = scenario.make_world()

    # post_step 콜백이 정의되어 있으면 사용 (매 스텝 후 실행되는 후처리 함수)
    if hasattr(scenario, 'post_step'):
        post_step = scenario.post_step
    else:
        post_step = None

    # 멀티에이전트 환경 생성
    if benchmark:
        # 벤치마크 모드: 추가 데이터 수집을 위한 콜백 포함
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            info_callback=scenario.benchmark_data,
                            discrete_action=discrete_action)
    else:
        # 일반 학습 모드
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            discrete_action=discrete_action)
    return env
