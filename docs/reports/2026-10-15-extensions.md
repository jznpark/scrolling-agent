# 확장 파라미터 3종 구현·시뮬레이션 보고서

- 작성일: 2026-10-15 · 작성: 박준호 (Claude Code로 구현)
- 대상: CSI6545 4조 scrolling-agent, 교수님 초안 v0.1 기준 MVP 환경에 사람에게서 보이는 현상 3가지를 추가
- 브랜치: `feature/value-error`, `feature/offscreen-residual`, `feature/attention-split` (모두 `feature/ppo` 위에서 따로 분기)

---

## 1. 요약

| # | 확장 | 새 설정 (기본값 = 꺼짐 = MVP와 동일) | 기대한 행동 | 시뮬레이션에서 확인된 것 |
|:-:|---|---|---|---|
| 1 | 가치 추정 오차 | `estimate.sigma_est`, `estimate.tau_learn`, `estimate.noise_lambda` | 열어 봤다 실망하고 바로 나가기(경우 ③), 너무 일찍 그만보기 | 오차가 클수록 짧은 시청(≤1초)이 9.5 → 15.3회로 늘고, 클릭한 후보가 실제보다 좋아 보였음(선택 편향). 보상 53.7 → 51.9, 천천히 배우면 48.9 |
| 3 | 화면 밖 잔여 처리 | `ddm.tau_residual`, `ddm.residual_in_watch` | 늦게 깨닫고 되돌아가기(경우 ②) | 화면 밖 이해가 에피소드당 0 → 3.7~6.2개, 되돌아가 클릭 0 → 1.8~3.0회. 보상은 소폭 상승(53.7 → 55.3), 대신 탐색 깊이 68 → 62 |
| 4 | 주의 분산 | `ddm.attention` (`none`/`log`/`power`), `ddm.attention_beta` | 화면이 붐비면 멈춰서 보는 시간이 늘어남 | 일정 속도 정책은 이해 수 59 → 40(log) → 19(power)로 급감. 가장 좋은 멈춤 시간이 0 → 0.25 → 0.5초, 목록에서 멈춘 시간이 0 → 16 → 26.5초로 증가 |

- 세 확장 모두 **꺼져 있으면 MVP와 결과가 완전히 같습니다**(난수까지 같음, 테스트로 확인). 따라서 합의된 설정은 바뀌지 않았습니다.
- 결과는 아직 **손으로 만든 비교 정책(baseline)** 기준입니다. 조건별 PPO 학습은 다음 단계입니다(6장).

---

## 2. 공통 준비 (세 브랜치가 함께 쓰는 기반, `feature/ppo` 커밋 `ac22f9d`, `db8f26a`)

| 추가한 것 | 내용 |
|---|---|
| `--set 섹션.필드=값` | 모든 스크립트에서 어떤 설정이든 명령줄로 변경 (예: `--set ddm.tau_residual=2.0`) |
| `scripts/simulate.py` | 조건(이름 + 설정 변경) × 정책을 **같은 목록·같은 DDM 난수**로 실행해 표와 CSV(`runs/sim/`) 출력 |
| `env.revealed_profile()` | 에이전트가 "알고 있는" A·λ. 비교 정책과 관측이 이 값만 사용 (확장 1에서 참값과 달라짐) |
| `env.extra_metrics()` | 확장별 지표를 에피소드 결과에 자동으로 붙이는 통로 |
| 공통 지표 | `n_short_bouts`(1초 이하 시청 = 열자마자 나감), `mean_bout`, `time_stay_list`(목록에서 멈춘 시간) |
| `GreedyKnown(dwell=…)` | "멈춰서 보기" 비교 정책: 보이는 미파악 후보가 `dwell`초 동안 노출될 때까지 멈춤 |

### 시뮬레이션 방법

- 목록: 학습에 쓰지 않은 평가용 seed 10000~10049 (50개). 모든 조건이 같은 목록을 씁니다.
- 정책: 비교 정책을 **고정**(시청 중단 기준 ρ = 0.6)하고 모델 조건만 바꿨습니다 = 초안 09장 실험 B("정책 고정 후 피드 변경")와 같은 방식. `--tune`을 주면 조건마다 ρ를 다시 맞춥니다(실험 A 근사).
  - `scrollfirst`: 일정 속도로 내려가며 처음 이해한 후보 중 보상률 ≥ ρ를 클릭, 보상률 < ρ면 back. 위로는 가지 않음
  - `greedy`: 아는 후보 중 보상률 최대인 곳으로(화면 밖이면 되돌아감) 이동해 시청, 없으면 아래로 탐색
  - `greedy-dwell`: greedy + 멈춰서 보기
- 기준(MVP) 결과: scrollfirst 53.65 ± 6.6, greedy 53.73 ± 6.6 (상한인 참조 해 104.4의 약 51%)

---

## 3. 확장 1 — 가치 추정 오차 (`feature/value-error`, 커밋 `3a53ef7`)

### 사람에게서 보이는 것
썸네일만 보고 판단하면 틀릴 때가 있다. 열어 보고 실망해 바로 나가거나, 좋은 것을 일찍 그만본다.

### 모델
이해가 끝날 때 공개되는 A·λ에 로그 정규 오차를 섞고, 시청하는 동안 오차가 사라진다.

    log Â = log A + σ_est · ε_A · f(w)        log λ̂ = log λ + σ_est · ε_λ · f(w)   (noise_lambda일 때)
    f(w) = exp(−w / τ_learn)    (τ_learn = 0이면 한 스텝 시청 후 바로 참값)

| 설정 | 기본 | 뜻 |
|---|---|---|
| `estimate.sigma_est` | 0.0 | 이해 시 로그 오차 크기 |
| `estimate.tau_learn` | 0.0 | 시청하며 오차가 사라지는 시간(초) |
| `estimate.noise_lambda` | True | λ에도 오차 |

### 기존과 다른 점
- **관측·정책**: 이해한 후보의 A·λ가 참값 대신 추정값으로 보임. 보상은 항상 참값으로 계산
- **난수**: 오차는 별도 난수 흐름(`SeedSequence.spawn(3)[2]`)에서 뽑아, 같은 행동이면 증거·이해 시점이 MVP와 완전히 같음 → 오차의 효과만 분리
- **새 지표**: `n_overestimated_clicks`(실제보다 좋아 보여 클릭한 횟수), `click_log_error`(클릭 시 log(추정/참) 평균, 양수 = 좋아 보이는 것만 고르는 선택 편향)
- **그림**: 요약 그림에 "인지한 A → 실제 A" 회색 선, 애니메이션에 참값을 빨간 글씨로 표시

### 결과 (50개 목록, ρ = 0.6)

| 조건 | 정책 | 보상 | 클릭 | 짧은 시청(≤1초) | 과대평가 클릭 | 클릭 시 로그 오차 |
|---|---|---|---|---|---|---|
| 오차 없음 (MVP) | greedy | 53.73 | 28.9 | 9.5 | 0 | 0.00 |
| σ_est 0.3 | greedy | 53.06 | 31.4 | 12.2 | 20.3 | +0.10 |
| σ_est 0.6 | greedy | 51.85 | 33.4 | 15.3 | 24.5 | +0.30 |
| σ_est 0.6, τ_learn 3초 | greedy | 48.89 | 31.0 | 10.5 | 22.9 | +0.32 |

(scrollfirst도 같은 경향: 53.65 → 53.01 → 51.56 → 48.76)

### 해석
- 오차가 클수록 **클릭이 늘고 짧은 시청이 늘어** 경우 ③(열어 봤다 바로 나가기)이 나타납니다.
- 클릭 시 로그 오차가 양수(+0.30): 기준을 넘는 것만 고르므로 **실제보다 좋아 보인 후보를 고르는 편향**이 생깁니다.
- 천천히 배우는 조건(τ 3초)은 짧은 시청은 다시 줄지만 보상 손실이 가장 큽니다. 나쁜 후보를 오래 보고, 좋은 후보를 나빠 보여 일찍 그만두기 때문입니다("너무 일찍 그만보기").

### 열린 점
- λ에도 오차를 넣을지, A에만 넣을지
- 초안의 "확률적인 실제 보상"은 아직 구현하지 않음

---

## 4. 확장 3 — 화면 밖 잔여 처리 (`feature/offscreen-residual`, 커밋 `207b32a`, `a57c720`) · 저희 가설

### 사람에게서 보이는 것
"아까 그거 괜찮았는데" 하고 늦게 깨닫는다.

### 모델
화면을 떠난 뒤에도 증거가 멈추지 않고, 떠난 뒤 지난 시간 s에 따라 줄어드는 비중으로 조금 더 쌓인다.

    가중치 a(s) = exp(−s / τ_p)
    한 구간 [s0, s1]의 유효 처리 시간 eff = τ_p (e^(−s0/τ_p) − e^(−s1/τ_p))
    dx = ν · g_last · eff + σ · √eff · N(0,1)

- 평균적으로 **떠날 때의 drift로 τ_p초만큼 더 처리**한 것과 같습니다(테스트로 평균·분산 확인).
- 화면 밖에서 임계값에 닿으면 화면 안과 똑같이 A·λ가 공개됩니다. 클릭은 강제하지 않습니다.

| 설정 | 기본 | 뜻 |
|---|---|---|
| `ddm.tau_residual` (τ_p) | 0.0 | 떠난 뒤 추가로 처리되는 평균 시간(초), 0 = 초안처럼 멈춤 |
| `ddm.residual_in_watch` | False | 시청·전환 중(목록이 가려졌을 때)에도 진행 |

### 기존과 다른 점
- **DDM**: 초안은 화면 밖 증거를 "멈춘 채 보존"합니다. 이 확장은 그 가정을 바꾸는 저희 제안입니다(기본은 꺼짐).
- **새 지표**:
  - `n_off_screen_understood`: 스크롤해 떠난 뒤 이해한 수
  - `n_hidden_understood`: 시청 중 가려진 상태에서 이해한 수
  - `n_late_clicks`: 늦게 이해한 후보로 되돌아가 클릭한 수
- **그림**: 경로 그림에 화면 밖 이해를 ★로 표시, 애니메이션에 "understood off screen: #i" 알림
- **비용**: 시뮬레이션 속도가 약 2배 느려짐(10 에피소드 1.1초 → 2.3초)

### 결과 (50개 목록, ρ = 0.6)

| 조건 | 정책 | 보상 | 이해 수 | 화면 밖 이해 | 되돌아가 클릭 | 가장 깊이 본 위치 |
|---|---|---|---|---|---|---|
| 멈춤 (MVP) | scrollfirst | 53.65 | 59.3 | 0 | 0 | 68.3 |
| 멈춤 (MVP) | greedy | 53.73 | 58.9 | 0 | 0 | 67.8 |
| τ_p 0.5초 | scrollfirst | 53.80 | 62.9 | 4.0 | 0 | 68.0 |
| τ_p 0.5초 | greedy | 55.28 | 58.7 | 3.7 | 1.8 | 63.7 |
| τ_p 2초 | scrollfirst | 53.49 | 65.8 | 6.8 | 0 | 68.4 |
| τ_p 2초 | greedy | 54.46 | 59.4 | 6.2 | 3.0 | 61.6 |
| τ_p 2초 + 시청 중에도 | greedy | 56.04 | 60.9 | 4.9 (+가려진 상태 18.3) | 2.5 | 60.6 |

### 해석
- 늦은 깨달음은 잔여 처리가 있을 때만 생기고, **경우 ②(되돌아가기)는 위로 갈 수 있는 정책(greedy)에서만** 나타납니다. scrollfirst는 더 많이 이해하지만 늦게 안 후보를 쓰지 못합니다.
- 되돌아가는 데 시간을 쓰므로 **탐색 깊이가 줄고**(68 → 62), 보상은 조금 오릅니다(+0.7~+1.6). "돌아갈까, 계속 탐색할까"의 절충이 생기며, 이는 학습된 정책이 풀어야 할 문제입니다.

### 열린 점 (교수님 확인 필요)
- 지수 감소 하나로 충분한지, 일정 시간 뒤 멈추거나 최근 몇 개만 처리해야 하는지(초안의 "기억 감소" 확장과 연결)
- 시청 중에도 진행할지(`residual_in_watch`). 초안의 "시청 중에는 목록 DDM 정지"와 충돌

---

## 5. 확장 4 — 주의 분산 (`feature/attention-split`, 커밋 `445b875`)

### 사람에게서 보이는 것
화면에 여러 개가 보이면 하나하나 파악이 느리다.

### 모델
보이는 미파악 콘텐츠 수 N에 따라 각 콘텐츠의 drift를 h(N)배로 줄인다(Kim 2025의 v₀/log₂N 방식).

    dx_i = ν_i · g(|v|) · h(N) dt + σ dW_i
    "log"  : h(N) = 1 / log₂(N + 1)   → h(1)=1, h(2)=0.63, h(3)=0.50, h(4)=0.43
    "power": h(N) = N^(−β)            → β=1이면 주의를 똑같이 나눔, h(3)=0.33

- Kim의 log₂N은 N=1에서 0으로 나누게 되므로 log₂(N+1)을 써서 **혼자 보이는 콘텐츠는 원래 속도**가 되게 했습니다.
- 이해가 끝난 콘텐츠는 주의를 차지하지 않으므로, 화면이 정리될수록 남은 콘텐츠가 빨라집니다(잡음이 없을 때 정확한 값으로 테스트).
- 잡음 σ는 초안대로 고정하고 drift만 나눕니다.

| 설정 | 기본 | 뜻 |
|---|---|---|
| `ddm.attention` | "none" | "none" / "log" / "power" |
| `ddm.attention_beta` | 1.0 | "power"의 지수 |

### 기존과 다른 점
- **DDM**: 초안은 "보이는 콘텐츠마다 독립적인 DDM"(합의)입니다. 이 확장은 drift 크기만 화면 상태에 따라 바꾸며, 콘텐츠별 독립 과정이라는 구조는 유지합니다.
- **새 지표**: `mean_unread_on_screen`(목록 시간 동안 보이는 미파악 수의 평균). 애니메이션 제목에 현재 N 표시
- **붐비는 화면 조건**: `screen.cards_per_screen=5`로 화면에 카드 5장. 속도 단위가 "화면 높이/초"라서, 카드 속도를 같게 하려면 `screen.speeds=(0.3,)`(0.3×5 = 1.5카드/초)를 함께 줍니다.

### 결과 1 — 정책 고정 (50개 목록, ρ = 0.6, dwell 1.5초)

| 조건 | 정책 | 보상 | 이해 수 | 목록에서 멈춘 시간(초) | 보이는 미파악 수 |
|---|---|---|---|---|---|
| 분산 없음 (MVP) | greedy | 53.73 | 58.9 | 0 | 1.94 |
| 분산 없음 (MVP) | greedy-dwell | 33.87 | 41.5 | 41.2 | 0.68 |
| log | greedy | 33.63 | 40.5 | 4.8* | 2.54 |
| log | greedy-dwell | 33.24 | 41.2 | 41.8 | 0.72 |
| power (β=1) | greedy | 6.61 | 18.7 | 32.8* | 1.99 |
| power (β=1) | greedy-dwell | 32.70 | 40.7 | 42.3 | 0.77 |
| 카드 5장 | greedy | 57.25 | 61.6 | 0 | 1.83 |
| 카드 5장 + log | greedy | 50.41 | 56.4 | 0 | 3.42 |

\* 목록 끝(99번)까지 내려가 거기서 기다린 시간

### 결과 2 — 조건별로 가장 좋은 멈춤 시간 (greedy, dwell 0~1.5초 탐색, 30개 목록)

| 주의 분산 | 가장 좋은 dwell | 그때 보상 | 목록에서 멈춘 시간(초) | 이해 수 |
|---|---|---|---|---|
| 없음 | 0초 | 53.5 | 0 | 59.8 |
| log | 0.25초 | 44.6 | 16.0 | 49.5 |
| power | 0.5초 | 39.7 | 26.5 | 44.6 |

### 해석
- 주의가 분산되면 일정 속도 스크롤로는 이해 수가 크게 줄어듭니다(59 → 40 → 19). 화면에 늘 2~3개의 미파악 후보가 있기 때문입니다.
- **멈추는 것이 최적이 됩니다.** 가장 좋은 멈춤 시간이 0 → 0.25 → 0.5초로 늘고, 목록에서 멈춘 총시간이 0 → 16 → 26.5초로 늘어납니다. 가설("화면이 붐비면 멈춰서 보는 시간이 늘어남")과 일치합니다.
- 화면에 카드가 많으면 분산이 없을 때는 이득(53.7 → 57.3)이지만, 분산이 있으면 그 이득이 대부분 사라집니다(50.4).

### 열린 점
- log와 power 중 어떤 형태가 맞는지, 잡음도 함께 나눌지(시간 분할 해석). 사람 데이터로 정해야 함
- 50% 미만으로 보이는 카드도 N에 넣을지

---

## 6. 한계와 다음 단계

**한계**
- 모든 결과는 **손으로 만든 비교 정책**이며, ρ·dwell을 고정했습니다. 조건에 맞춰 최적화하는 에이전트(PPO)라면 효과의 크기가 달라질 수 있습니다. 특히 확장 1(오차를 감안해 덜 믿기)과 확장 3(돌아갈지 판단)은 학습된 정책에서 더 분명해질 것으로 예상합니다.
- 목록 50개, 시드 1세트 결과입니다. 수치(σ_est, τ_p, β 등)는 모두 [TBD] 임시값입니다.
- 1M 스텝 PPO 시범 학습도 아직 비교 정책보다 낮습니다(상한 대비 16~27% vs 51%). 확장 조건 학습 전에 기본 PPO 개선이 먼저 필요합니다.

**다음 단계 제안**
1. 기본 PPO 개선(오래 학습 + `ent_coef` 낮추기)
2. 조건별 PPO 학습. 예: `python scripts\train_ppo.py --name att_log --set ddm.attention=log`
3. 학습된 정책을 같은 목록에서 평가: `python scripts\simulate.py --cond log ddm.attention=log --policies greedy ppo:runs\att_log`
4. 세 브랜치 통합: `env.py`에서만 충돌이 나고(가치 오차와 잔여 처리가 같은 `extra_metrics` 함수와 설명 문단을 수정), 손으로 쉽게 해결할 수 있습니다. 통합 후에는 조건을 섞은 실험(예: 오차 + 잔여 처리)이 가능합니다.
5. 교수님 확인: 잔여 처리의 형태·시청 중 진행 여부, 주의 분산 형태, 추정 오차를 λ에도 넣을지

---

## 7. 재현 방법 (PowerShell, venv 활성화)

```powershell
# 확장 1
git switch feature/value-error
python scripts\simulate.py --cond exact --cond est0.3 estimate.sigma_est=0.3 --cond est0.6 estimate.sigma_est=0.6 --cond est0.6_slow estimate.sigma_est=0.6 estimate.tau_learn=3 --policies scrollfirst greedy --episodes 50
python scripts\visualize.py --policy greedy --rho 0.6 --set estimate.sigma_est=0.6 --animate

# 확장 3
git switch feature/offscreen-residual
python scripts\simulate.py --cond frozen --cond tau0.5 ddm.tau_residual=0.5 --cond tau2 ddm.tau_residual=2.0 --cond tau2_watch ddm.tau_residual=2.0 ddm.residual_in_watch=True --policies scrollfirst greedy --episodes 50
python scripts\visualize.py --policy greedy --rho 0.6 --set ddm.tau_residual=2.0 --animate

# 확장 4
git switch feature/attention-split
python scripts\simulate.py --cond base --cond log ddm.attention=log --cond power1 ddm.attention=power --cond crowd5 screen.cards_per_screen=5 "screen.speeds=(0.3,)" --cond crowd5_log screen.cards_per_screen=5 "screen.speeds=(0.3,)" ddm.attention=log --policies scrollfirst greedy greedy-dwell --episodes 50
python scripts\visualize.py --policy greedy --dwell 0.5 --rho 0.6 --set ddm.attention=power --animate
```

각 브랜치의 상세 설명: `docs/extensions/value-error.md`, `offscreen-residual.md`, `attention-split.md`

| 브랜치 | 테스트 |
|---|---|
| `feature/ppo` (공통 기반) | 81개 통과 |
| `feature/value-error` | 91개 통과 |
| `feature/offscreen-residual` | 91개 통과 |
| `feature/attention-split` | 94개 통과 |
