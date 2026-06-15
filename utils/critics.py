"""
어텐션 크리틱 모듈
MAAC의 핵심 구성 요소로, 멀티헤드 어텐션 메커니즘을 사용하는 중앙 크리틱 네트워크

구조 개요:
- 각 에이전트는 자신의 상태-행동 인코딩을 가짐
- 어텐션 메커니즘으로 다른 에이전트들의 정보를 선택적으로 집계
- 자신의 상태 인코딩 + 어텐션 결과를 합쳐 Q값을 계산

이 구조의 장점:
1. 에이전트 수에 유연하게 대응 가능
2. 중요한 에이전트의 정보에 더 많은 가중치를 부여
3. 해석 가능한 어텐션 가중치 제공
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain  # 여러 이터러블을 하나로 연결


class AttentionCritic(nn.Module):
    """
    어텐션 기반 크리틱 네트워크
    모든 에이전트가 공유하는 중앙 크리틱으로, 멀티헤드 어텐션을 통해
    다른 에이전트들의 상태-행동 정보를 동적으로 집계하여 Q값을 추정함

    'Attention is All You Need' 논문의 스케일드 닷-프로덕트 어텐션을 사용

    네트워크 구성:
        - critic_encoders: 상태+행동을 은닉 표현으로 인코딩 (에이전트 공유)
        - state_encoders: 상태만 인코딩 (셀렉터 생성용)
        - key_extractors: 어텐션 키(Key) 추출 (헤드별)
        - selector_extractors: 어텐션 쿼리(Query) 추출 (헤드별)
        - value_extractors: 어텐션 값(Value) 추출 (헤드별)
        - critics: 최종 Q값 출력 네트워크 (에이전트별)
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        인자:
            sa_sizes (list of (int, int)): 각 에이전트의 (상태 차원, 행동 차원) 튜플 리스트
            hidden_dim (int): 은닉층 차원 (기본값: 32)
            norm_in (bool): 입력 배치 정규화 적용 여부 (기본값: True)
            attend_heads (int): 멀티헤드 어텐션의 헤드 수 (기본값: 1)
                hidden_dim이 attend_heads로 나누어 떨어져야 함
        """
        super(AttentionCritic, self).__init__()
        # 은닉 차원이 어텐션 헤드 수로 나누어 떨어지는지 확인
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)  # 에이전트 수
        self.attend_heads = attend_heads  # 어텐션 헤드 수

        self.critic_encoders = nn.ModuleList()  # 상태-행동 인코더 리스트
        self.critics = nn.ModuleList()  # Q값 출력 네트워크 리스트

        self.state_encoders = nn.ModuleList()  # 상태 인코더 리스트 (셀렉터/쿼리 용)
        # 각 에이전트별 네트워크 생성
        for sdim, adim in sa_sizes:
            idim = sdim + adim  # 인코더 입력 차원 = 상태 + 행동
            odim = adim  # 크리틱 출력 차원 = 행동 수 (각 행동의 Q값)

            # 상태-행동 인코더: [BatchNorm →] Linear → LeakyReLU
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            # Q값 출력 네트워크: Linear → LeakyReLU → Linear
            # 입력: 자신의 상태 인코딩 + 어텐션 결과 (2 * hidden_dim)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            # 상태 인코더: 자신의 상태만 인코딩 (어텐션 쿼리 생성에 사용)
            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        # === 멀티헤드 어텐션 구성 요소 ===
        attend_dim = hidden_dim // attend_heads  # 각 헤드의 차원 = 은닉 차원 / 헤드 수
        self.key_extractors = nn.ModuleList()  # 키(Key) 추출기: 다른 에이전트의 정보를 표현
        self.selector_extractors = nn.ModuleList()  # 셀렉터(Query) 추출기: 자신이 어떤 정보에 주의를 기울일지 결정
        self.value_extractors = nn.ModuleList()  # 값(Value) 추출기: 어텐션 가중치로 집계할 실제 정보
        for i in range(attend_heads):
            # 키: 편향(bias) 없는 선형 변환
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            # 셀렉터(쿼리): 편향 없는 선형 변환
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            # 값: 선형 변환 + LeakyReLU 활성화
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        # 에이전트 간 공유되는 모듈 리스트 (그래디언트 스케일링에 사용)
        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        에이전트 간 공유되는 파라미터를 반환
        키, 셀렉터, 값 추출기 및 크리틱 인코더의 파라미터

        반환:
            chain: 모든 공유 파라미터의 이터레이터
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        공유 파라미터의 그래디언트를 에이전트 수로 나누어 스케일링

        공유 모듈은 각 에이전트의 크리틱 손실에서 그래디언트가 중복 누적되므로
        에이전트 수(nagents)로 나누어 정확한 평균 그래디언트를 계산
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        순전파: 모든 에이전트의 상태-행동 쌍을 입력받아 Q값을 계산

        핵심 과정:
        1. 각 에이전트의 상태-행동을 인코딩
        2. 멀티헤드 어텐션으로 다른 에이전트 정보 집계
        3. 자신의 상태 + 어텐션 결과로 Q값 계산

        인자:
            inps (list of tuples): [(상태, 행동), ...] 각 에이전트의 입력
            agents (range/list): Q값을 계산할 에이전트 인덱스 (None이면 전체)
            return_q (bool): Q값 반환 여부 (선택된 행동의 Q)
            return_all_q (bool): 모든 행동의 Q값 반환 여부
            regularize (bool): 어텐션 로짓 정규화 항 반환 여부
            return_attend (bool): 어텐션 가중치 반환 여부
            logger (SummaryWriter): TensorBoard 로거
            niter (int): 현재 반복 횟수 (로깅용)
        반환:
            Q값 또는 [Q값, 전체Q, 정규화항, 어텐션가중치] 등 요청된 값의 리스트
        """
        if agents is None:
            agents = range(len(self.critic_encoders))  # 기본: 모든 에이전트
        # 입력에서 상태와 행동 분리
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        # 상태와 행동을 연결하여 인코더 입력 생성
        inps = [torch.cat((s, a), dim=1) for s, a in inps]

        # === 인코딩 단계 ===
        # 각 에이전트의 상태-행동을 은닉 표현으로 인코딩
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # Q값을 계산할 에이전트들의 상태 인코딩 (어텐션 쿼리 생성용)
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]

        # === 어텐션 키/값/쿼리 추출 ===
        # 각 헤드별로 모든 에이전트의 키(Key) 추출
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # 각 헤드별로 모든 에이전트의 값(Value) 추출
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # 각 헤드별로 Q값을 계산할 에이전트의 셀렉터(Query) 추출
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        # 어텐션 결과를 저장할 리스트 초기화
        other_all_values = [[] for _ in range(len(agents))]  # 집계된 다른 에이전트 정보
        all_attend_logits = [[] for _ in range(len(agents))]  # 어텐션 로짓 (정규화용)
        all_attend_probs = [[] for _ in range(len(agents))]  # 어텐션 가중치 (시각화용)

        # === 멀티헤드 어텐션 계산 ===
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # 각 에이전트에 대해 어텐션 수행
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                # 자기 자신을 제외한 다른 에이전트의 키와 값 수집
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # 스케일드 닷-프로덕트 어텐션 계산
                # 셀렉터(쿼리)와 키의 내적으로 어텐션 점수 계산
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # 키 차원의 제곱근으로 스케일링 (Attention is All You Need)
                # 내적 값이 너무 커지는 것을 방지하여 소프트맥스의 그래디언트 소실 완화
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                # 소프트맥스로 어텐션 가중치 정규화 (합 = 1)
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                # 가중치를 적용하여 다른 에이전트들의 값을 가중 합산
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # === Q값 계산 ===
        all_rets = []
        for i, a_i in enumerate(agents):
            # 각 헤드의 어텐션 엔트로피 계산 (어텐션이 얼마나 분산되어 있는지 측정)
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            # 자신의 상태 인코딩 + 모든 헤드의 어텐션 결과를 연결하여 크리틱 입력 생성
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            # 크리틱 네트워크로 모든 행동에 대한 Q값 계산
            all_q = self.critics[a_i](critic_in)
            # 실제 선택된 행동의 인덱스 추출
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            # 선택된 행동에 해당하는 Q값만 추출
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)  # 선택된 행동의 Q값
            if return_all_q:
                agent_rets.append(all_q)  # 모든 행동의 Q값 (정책 업데이트에 사용)
            if regularize:
                # 어텐션 로짓의 크기를 정규화하여 어텐션이 너무 극단적이 되는 것을 방지
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))  # 어텐션 가중치 (분석/시각화용)
            if logger is not None:
                # 각 헤드의 어텐션 엔트로피를 TensorBoard에 기록
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            # 반환값이 하나면 리스트 없이 직접 반환
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        # 에이전트가 하나면 리스트 없이 직접 반환
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
