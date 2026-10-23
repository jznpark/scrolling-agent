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

# 4) 조건 비교 시뮬레이션 — 조건(이름 + 설정 변경) x 정책, 같은 목록에서
python scripts\simulate.py --cond base --cond crowded screen.cards_per_screen=5
python scripts\simulate.py --cond base --cond kappa08 feed.kappa=0.8 --tune     # 조건마다 rho 다시 맞춤
```

### 웹 화면에서 실시간으로 보기 (로컬 서버, 추가 설치 없음)

```powershell
python scripts\serve.py --open                  # 현재 브랜치: http://127.0.0.1:8000
python scripts\serve_branches.py --open         # 브랜치 4개를 탭으로 (모두 8000 하나로: /b/1/ ~ /b/4/)
python scripts\serve_branches.py --branches feature/value-error origin/feature/attention-split
```

- 정책(random / scrollfirst / greedy / greedy-dwell / 학습된 PPO / **human = 직접 플레이**)과 seed, 모든 설정을 화면에서 바꾸고 재생·일시정지·한 스텝
- 직접 플레이: `↓`/`↑` 누르고 있으면 이동, `1`~`4` 보이는 칸 클릭, `B` back, `Space` 멈춤
- `H` 숨은 값(실제 A·ν·z) 표시, 오른쪽에 전체 목록 지도·상태·누적 보상·사건 기록(늦은 이해 등)
- 주소로 조건 지정: `http://127.0.0.1:8000/?policy=greedy-dwell&dwell=0.5&set=ddm.attention=log&skip=300&autoplay=1`
- `serve_branches.py`는 각 브랜치를 `.worktrees/`에 따로 꺼내 실행하므로 브랜치 자체는 바뀌지 않습니다
- 설정마다 `?`에 마우스를 올리면 쉬운 설명이 나옵니다 (설명·쉬운 이름·상황 목록: `webdemo/descriptions.py`)
- **상황 고르기**: "좋은 콘텐츠가 드문 피드", "금방 질리는 콘텐츠", "아까 그거 괜찮았는데" 등을 고르면 설정이 바뀌고,
  아래 **행동 비교**가 기본 설정과 같은 피드 20개로 돌려 "열자마자 닫은 횟수 3배 ↑"처럼 달라진 행동을 보여 줍니다
- 비교 페이지(8000)는 위쪽 **탭**으로 브랜치를 바꿔 봅니다 (주소 끝 `#2`처럼 탭 번호 지정 가능)
- **늦은 깨달음 보기**: '늦은 깨달음' 탭 → 상황 '아까 그거 괜찮았는데' → '중요한 순간에 멈춤' 체크 → ▶. 지나간 카드를 뒤늦게 알아챈 순간과 그 카드로 되돌아와 여는 순간에 멈추고, 그 카드에는 ★가 붙습니다
- 주소로 바로 열기: `http://127.0.0.1:8001/?scenario=boring_fast` (상황 id는 `descriptions.py` 참고)
- **그만 보기 알림 [확장, 기본 꺼짐]**: 상황 "30초 뒤 그만 보기 알림"에서 계속 보기·종료를 선택할 수 있습니다.
  설정·재현 방법·검증 결과는 [확장 설명](docs/extensions/break-reminder.md)을 참고하세요.

#### 다른 PC에서 접속하기

비교 페이지는 **포트 8000 하나**로 모든 브랜치를 보여 줍니다(`/b/1/`, `/b/2/` …로 중계). 밖에 열 포트는 8000 하나입니다.

**A. 다른 곳(다른 네트워크)에 있는 팀원 — Cloudflare 임시 터널 (추천)**

`192.168.x.x` 같은 사설 IP는 같은 공유기 안에서만 열리므로, 다른 곳에서는 터널이 필요합니다.

```powershell
winget install --id Cloudflare.cloudflared               # 처음 한 번
python scripts\serve_branches.py --token 비밀번호          # 터미널 1 (이 PC에서만 열어도 됨)
cloudflared tunnel --url http://localhost:8000             # 터미널 2 → https://xxxx.trycloudflare.com 주소가 나옴
```

- 팀원에게 `https://xxxx.trycloudflare.com/?token=비밀번호` 를 보냅니다. 한 번 들어오면 쿠키로 기억합니다.
- 주소는 실행할 때마다 바뀌고, 두 터미널을 끄면 바로 닫힙니다. 쓰는 동안에만 켜 두세요.
- `--token` 없이 열면 링크를 아는 누구나 들어올 수 있습니다.

**B. 같은 공유기(같은 와이파이)에 있는 PC**

```powershell
python scripts\serve_branches.py --host 0.0.0.0
# "관리자 권한" PowerShell에서 한 번만: 포트 8000 열기
New-NetFirewallRule -DisplayName "scrolling-agent demo" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Domain
Get-NetConnectionProfile                                   # Public이면 규칙이 적용되지 않음 → 개인 네트워크로 변경
Remove-NetFirewallRule -DisplayName "scrolling-agent demo" # 다 쓴 뒤
```

- 콘솔에 나온 주소(예: `http://192.168.0.3:8000`)로 접속합니다. 처음 실행 때 "Python 액세스 허용" 창이 뜨면 **개인 네트워크**만 허용합니다.
- 학교 와이파이(eduroam 등)는 기기끼리 연결을 막는 경우가 많습니다 → A 방법을 쓰세요.
- 다른 대안: VS Code 하단 **포트** 탭에서 8000을 전달(공개 범위 Public), 또는 팀원 모두 Tailscale 설치.

어떤 설정이든 `--set 섹션.필드=값`으로 바꿀 수 있습니다 (예: `--set ddm.noise=0.3 screen.speeds="(0.5, 1.5)"`).
섹션: `time`, `screen`, `ddm`, `feed` (+ 확장 브랜치의 새 섹션).

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
