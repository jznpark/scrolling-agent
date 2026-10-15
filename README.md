# Scrolling Agent

제한된 시간 안에서 피드를 스크롤하며 **탐색(썸네일 파악) · 소비(시청) · 재방문** 사이에 시간을 배분하는 에이전트를,
계산적 합리성(Computational Rationality) 관점에서 DDM + 강화학습(PPO)으로 구현하는 프로젝트입니다.

- 과목: CSI6545 Computational Interaction Design (연세대, 2026 가을) · 4조
- 팀: 박준호 · 김수민 · 황도현 · 박세은 · 백하늘
- 설계 기준: 교수님 초안 *Scrolling Agent — 초기 연구 설계* v0.1 (2026-10-14)

## 핵심 아이디어

| 구성 | 내용 |
|---|---|
| 피드 | 1차원 목록의 콘텐츠 `C_i = (ν_i, A_i, λ_i)` — 이해 속도, 초기 보상률, 보상 감소율 |
| 지각 (DDM) | 화면에 보이는 콘텐츠마다 독립적인 증거 누적. 임계값 `B`에 처음 닿으면 `A_i, λ_i` 공개 (클릭은 강제되지 않음) |
| 속도 효과 | 빠르게 스크롤할수록 drift 감소: `g(|v|) = 1 / (1 + (|v|/v0)^pv)` |
| 보상 | 시청할 때만 보상률 `A_i · exp(-λ_i · w_i)`의 적분. 매 스텝 고정 비용 `c` |
| 목표 | 고정 세션 시간 `T` 안의 누적 보상 최대화 (γ = 1) |
| 행동 | `down(speed)`, `up(speed)`, `stay(None | i)`, `back` · 목록 모드 / 시청 모드 |
| 학습 | Gymnasium 환경 + Stable-Baselines3 PPO (행동 마스킹: sb3-contrib MaskablePPO) |

## 폴더 구조

```
scrolling-agent/
├── src/scrolling_agent/
│   ├── config.py      # 환경·DDM·피드 설정값 (초안의 시작용 구성)
│   ├── feed.py        # 확률적 피드 생성기
│   ├── env.py         # Gymnasium 환경
│   ├── policies.py    # 비교 정책, PPO 정책 래퍼, 참조 해(상한)
│   ├── evaluation.py  # 에피소드 실행·지표·기록
│   └── viz.py         # 에피소드 그림·애니메이션
├── scripts/
│   ├── check_setup.py # 설치 확인: 라이브러리 import + PPO 짧은 학습
│   ├── train_ppo.py   # MaskablePPO 학습
│   ├── evaluate.py    # PPO vs 비교 정책 평가
│   └── visualize.py   # 한 에피소드 보기
├── tests/             # pytest
└── docs/
    └── design-notes.md
```

## 설치 (Windows · PowerShell 기준)

Python 3.10 이상이 필요합니다.

```powershell
cd C:\Workspace\scrolling-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/check_setup.py
pytest
```

> PyTorch는 기본으로 CPU 버전이 설치됩니다. GPU를 쓰려면 https://pytorch.org 에서 CUDA 버전 설치 명령을 먼저 실행하세요.

## 실행 방법 (PowerShell, venv 활성화 후)

```powershell
# 1) 한 에피소드 눈으로 보기 — 정책: random / scrollfirst / greedy / ppo
python scripts\visualize.py --policy greedy --rho 0.6              # 요약 그림 창
python scripts\visualize.py --policy greedy --rho 0.6 --animate    # 재생 (스페이스 = 일시정지)
python scripts\visualize.py --policy ppo --run runs\pilot --animate
python scripts\visualize.py --policy greedy --kappa 0.8 --d-nu -0.7  # 다른 피드 조건
python scripts\visualize.py --policy ppo --run runs\pilot --save runs\pilot\ep.png   # 파일로 저장 (.png / .gif)

# 2) PPO 학습 (결과: runs\<name>\ — git에 올라가지 않음)
python scripts\train_ppo.py --name pilot --timesteps 1000000 --seed 0
tensorboard --logdir runs                                          # 학습 곡선: http://localhost:6006

# 3) 비교 평가 — 같은 미사용 목록 100개에서 PPO와 비교 정책
python scripts\evaluate.py --runs runs\pilot
```

`--seed`는 목록과 DDM 잡음을 고릅니다. 10000번 이상은 학습에 쓰지 않은 평가용 목록입니다.
그림의 z 색과 아직 이해하지 못한 후보의 A·ν는 에이전트에게 보이지 않는 값이며, 분석용으로만 표시합니다.

## 진행 단계

- [x] 0. 저장소·프로젝트 설정
- [x] 1. 피드 생성기 (`feed.py`) — seed로 재현 가능, 같은 집합의 순서 변경
- [x] 2. 환경 (`env.py`) — 고정 시간, 가시성, DDM, 정보 공개, 시청 보상, 상태 전환
- [x] 3. 구현 검증 — 평균 이해시간 ≈ B/ν, 시간 합계 = T, 정보 보존
- [x] 4. 비교 정책 (`policies.py`) — 일정 속도 + 첫 이해 후보 선택, 알려진 후보 중 최대 구간 보상
- [ ] 5. PPO 학습·평가 — 여러 seed, 학습에 쓰지 않은 목록으로 평가
- [ ] 6. 피드 조건 실험 — dν(유망 후보의 파악 난이도), κ(모이는 정도)

## 브랜치 규칙 (제안)

- `main`: 항상 실행되는 상태 유지
- 작업은 `feature/<내용>` 브랜치에서 하고 Pull Request로 합칩니다
