# 추가 RL 테스트 A 묶음 — 진행 현황

- 작성: 2026-11-11 · 박준호 (Claude Code로 실행)
- 브랜치: `feature/rl-tests-a` (`main` 9a84031에서 시작)
- 근거: `03-PPO-스크롤속도-결과보고.md`(11-09)에서 남은 질문
- **상태: 코드 완료, 학습 진행 중. 아래 표의 결과 칸은 아직 비어 있습니다.**

## 1. 테스트 목록과 상태

| # | 테스트 | 조건 이름 | 확인할 것 | 상태 |
|:-:|---|---|---|:-:|
| A1 | 화면 밖 잔여 처리 + PPO | `resid-0.5`, `resid-2` (대조군 `a-base`) | PPO도 뒤늦게 이해한 아이템으로 되돌아가는가 | 🟨 학습 중 |
| A2 | 한 속도 PPO, 같은 코드 | `a-base`, `a-base-k0.9` | 속도 선택지를 추가한 효과 | ⬜ 학습 대기 |
| A3 | 두 속도 κ 0, 2M | `speed2-2m` | κ 0.9 2M(11-09)과 학습량 맞추기 | ⬜ 학습 대기 |
| A4 | 같은 아이템의 순서만 변경 | `reorder-k0`, `reorder-k0.9` | κ별로 피드가 달라서 생긴 차이를 제거 | ⬜ 학습 대기 |
| A5 | 결정 시점 기준 늦추기 지수 | `speed2`, `speed2-k0.9` 모델 분석 | 유지 중인 스텝을 뺀 지수 | ⬜ 학습 대기 |

**공통 학습 설정**: topk 관측, 결정 간격 0.5초, 학습률 3e-4 → 0 감소, 결정 1M번(`speed2-2m`만 2M), seed 3개, 최고 모델 선택은 평가와 겹치지 않는 피드 10개(seed 30000~). 평가는 피드 100개(seed 10000~).

**학습 순서**: 6개씩 차례로 → (1) `resid-0.5` `resid-2` (2) `speed2` `speed2-k0.9` (3) `a-base` `a-base-k0.9` (4) `speed2-2m` `reorder-k0` (5) `reorder-k0.9`. 시작 11-11 16:16, 예상 약 2시간 30분(11-05 실측 기준, 두 속도 조건은 이 PC에서 처음이라 달라질 수 있음).

## 2. 이번에 추가한 코드 (커밋 346bc90)

| 파일 | 내용 |
|---|---|
| `scripts/pipeline.py` | 위 조건 9개. 결과표에 `revisit_clicks` 열 추가 |
| `src/scrolling_agent/config.py`, `env.py` | `feed.reorder_kappa`: 만든 아이템은 그대로 두고 순서만 바꿈. 기본은 꺼짐(`None`)이라 기존 결과는 변하지 않음 |
| `scripts/experiments/speed_context.py` | 늦추기 지수 5종: 전체 / 새 결정만 / 이해 직후 첫 결정만 / 1초 이내 / 동시 이해 제외 |
| `tests/test_rl_tests_a.py` | 순서 변경, 조건 생성, 지수 부호·제외 규칙 검사 |
| `webdemo/descriptions.py` | 새 설정의 한국어 설명 |

테스트 233개 통과. 두 속도 + 순서 변경 조건으로 짧은 학습과 지수 계산까지 끝까지 도는 것을 확인했습니다.

## 3. 알아 둘 점

- **`feature/ppo-speed-context`(296565b)가 GitHub에 없습니다.** 11-09 실험의 코드·모델을 받을 수 없어서, 두 속도 모델을 이 PC에서 새로 학습하고 늦추기 지수를 보고서의 정의대로 다시 구현했습니다. 같은 스텝에 여러 아이템을 이해한 경우의 처리(여기서는 목록에서 가장 아래 아이템 기준)는 11-09 구현과 다를 수 있습니다.
- `speed2`, `speed2-k0.9`가 11-09 수치(늦추기 지수 0.014 / 0.169)와 비슷하게 나오는지가 재구현 확인이 됩니다.
- 두 속도 (0.5, 1.5)는 초안에 "나중에"로 적힌 값입니다. 합의 설정(기본값)은 바꾸지 않았고, 조건별로만 켭니다.

## 4. 끝나면 할 일

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\pipeline.py evaluate resid-0.5 resid-2 a-base a-base-k0.9 speed2 speed2-k0.9 speed2-2m reorder-k0 reorder-k0.9 --episodes 100
python scripts\pipeline.py table resid-0.5 resid-2 a-base a-base-k0.9 speed2 speed2-k0.9 speed2-2m reorder-k0 reorder-k0.9
python scripts\experiments\speed_context.py --runs runs\speed2 runs\speed2-k0.9 runs\speed2-2m runs\reorder-k0 runs\reorder-k0.9 --episodes 100
```

중단되면 같은 `train` 명령에 `--resume`을 붙여 이어 갑니다.
