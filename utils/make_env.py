"""환경 생성 (현재는 EV 충전 환경만 지원)."""

def make_env(scenario_name, benchmark=False, discrete_action=False, arrival_mode='normal'):
    if scenario_name == 'ev_charging':
        from envs.ev_charging import EVChargingEnv
        return EVChargingEnv(arrival_mode=arrival_mode)

    # MPE(multi-agent particle env) 시나리오 (원본 MAAC 호환용; 본 저장소에선 미사용)
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as old_scenarios
    import envs.mpe_scenarios as new_scenarios

    try:
        scenario = old_scenarios.load(scenario_name + ".py").Scenario()
    except:
        scenario = new_scenarios.load(scenario_name + ".py").Scenario()

    world = scenario.make_world()
    post_step = scenario.post_step if hasattr(scenario, 'post_step') else None

    if benchmark:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            info_callback=scenario.benchmark_data,
                            discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            discrete_action=discrete_action)
    return env
