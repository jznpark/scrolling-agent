"""Plain-language names, "?" explanations and ready-made situations for the web demo.

Written for people who do not know the model: everyday words, no jargon.
Keys are "section.field"; settings that exist only on some branches are listed too
(they are shown only when the running branch has them).
"""

import math

# Short everyday name shown next to each setting.
LABELS: dict[str, str] = {
    "reminder.enabled": "그만 보기 알림 켜기",
    "reminder.after_seconds": "처음 알릴 사용 시간 (초)",
    "reminder.snooze_seconds": "계속 보기 후 다시 알릴 간격 (초)",
    "time.session_seconds": "전체 사용 시간",
    "time.dt": "한 번 판단하는 간격",
    "time.step_cost": "시간이 아까운 정도",
    "time.transition_seconds": "열고 닫는 데 걸리는 시간",
    "screen.n_items": "목록에 있는 콘텐츠 수",
    "screen.cards_per_screen": "한 화면에 보이는 카드 수",
    "screen.visible_fraction": "얼마나 보여야 '봤다'고 치나",
    "screen.speeds": "스크롤 속도",
    "ddm.threshold": "알아차리는 기준",
    "ddm.noise": "알아차리는 시간의 들쭉날쭉함",
    "ddm.v0": "빨리 넘기면 못 알아보는 정도",
    "ddm.pv": "속도에 따른 변화의 급함",
    "ddm.dt_internal": "계산 정밀도",
    "ddm.tau_residual": "지나간 뒤에도 생각나는 시간",
    "ddm.residual_in_watch": "보는 중에도 지나간 것 생각",
    "ddm.attention": "화면이 붐비면 헷갈림",
    "ddm.attention_beta": "헷갈림의 세기",
    "feed.p": "좋은 콘텐츠 비율",
    "feed.kappa": "좋은 것끼리 몰려 있는 정도",
    "feed.a0": "보통 콘텐츠의 재미",
    "feed.a1": "좋은 콘텐츠가 더 재미있는 정도",
    "feed.sigma_A": "재미의 들쭉날쭉함",
    "feed.m_nu": "썸네일 알아보기 쉬운 정도",
    "feed.d_nu": "좋은 콘텐츠의 썸네일 난이도",
    "feed.sigma_nu": "썸네일 난이도의 들쭉날쭉함",
    "feed.m_lambda": "금방 질리는 정도",
    "feed.sigma_lambda": "질리는 속도의 들쭉날쭉함",
    "estimate.sigma_est": "썸네일만 보고 착각하는 정도",
    "estimate.noise_lambda": "얼마나 오래 재밌을지도 착각",
    "estimate.tau_learn": "보면서 착각을 깨닫는 시간",
}

# "?" explanations: what it is in daily life, and what happens when you raise it.
DESCRIPTIONS: dict[str, str] = {
    "reminder.enabled": "설정한 시뮬레이션 시간이 되면 재생을 멈추고 계속 볼지, 그만 볼지 묻습니다. "
                        "기본값은 꺼짐입니다. 실제 벽시계 시간이나 콘텐츠 시청 시간만 세는 기능은 아닙니다.",
    "reminder.after_seconds": "목록 탐색·콘텐츠 시청·전환을 모두 합친 시뮬레이션 사용 시간입니다. "
                              "처음 이 시간에 도달한 스텝에서 알립니다. 세션 종료 시각 이후로 설정하면 알리지 않습니다.",
    "reminder.snooze_seconds": "'계속 보기'를 선택한 뒤 이만큼 시뮬레이션 시간이 더 지나면 다시 묻습니다. "
                               "알림에 답하는 동안 시뮬레이션은 진행되지 않습니다.",
    # time
    "time.session_seconds": "앱을 쓰는 전체 시간(초)입니다. 예를 들어 120이면 \"2분만 볼게\" 하고 켠 상황입니다. "
                            "무엇을 하든 시간은 흘러갑니다.",
    "time.dt": "사람이 \"계속 볼까, 넘길까\"를 다시 생각하는 간격(초)입니다. 보통은 바꿀 필요가 없습니다.",
    "time.step_cost": "아무것도 얻지 못하고 시간만 가는 것이 얼마나 아까운지입니다. "
                      "모든 순간에 똑같이 깎이기 때문에, 이 값만 바꿔서는 행동이 거의 바뀌지 않습니다(전체 점수만 내려감).",
    "time.transition_seconds": "콘텐츠를 눌러 여는 데, 그리고 닫고 목록으로 돌아오는 데 걸리는 시간(초)입니다. "
                               "길수록 이것저것 열어 보기가 부담스러워집니다.",
    # screen
    "screen.n_items": "피드에 있는 콘텐츠 개수입니다. 끝까지 내려가면 더 볼 것이 없습니다.",
    "screen.cards_per_screen": "휴대폰 한 화면에 카드가 몇 장 보이는지입니다. 많을수록 한 번에 여러 개를 훑어볼 수 있습니다.",
    "screen.visible_fraction": "카드가 화면에 얼마나 보여야 \"봤다\"고 치는지입니다. 0.5면 절반 이상 보여야 눈에 들어옵니다.",
    "screen.speeds": "손가락으로 넘기는 속도입니다(1초에 화면 몇 개만큼). 빠를수록 많이 지나가지만 하나하나는 제대로 못 봅니다. "
                     "쉼표로 두 개(예: 0.5, 1.5)를 넣으면 천천히/빠르게를 골라 쓸 수 있습니다.",
    # ddm
    "ddm.threshold": "썸네일을 보고 \"아, 이건 이런 거구나\" 하고 알아차리려면 얼마나 확신이 필요한지입니다. "
                     "높을수록 알아차리는 데 오래 걸리고, 알아차려야 재미가 얼마나 될지 알 수 있습니다.",
    "ddm.noise": "같은 썸네일이라도 금방 알아볼 때와 한참 걸릴 때가 있는 정도입니다. 클수록 운에 따라 들쭉날쭉합니다.",
    "ddm.v0": "얼마나 빨리 넘기면 썸네일을 잘 못 알아보게 되는지입니다. 작을수록 조금만 빨리 넘겨도 눈에 안 들어옵니다.",
    "ddm.pv": "넘기는 속도가 빨라질 때 알아보기 어려워지는 모양입니다. 클수록 어느 속도를 넘으면 갑자기 안 보입니다.",
    "ddm.dt_internal": "화면이 움직이는 동안 얼마나 촘촘하게 계산할지입니다. 작을수록 정확하지만 느립니다. 보통은 그대로 두세요.",
    "ddm.tau_residual": "지나간 카드를 화면에서 넘긴 뒤에도 머릿속으로 계속 생각하는 시간(초)입니다. "
                        "0이면 넘기는 순간 생각도 멈춥니다. 키우면 \"아까 그거 괜찮았는데?\" 하고 늦게 깨닫고 "
                        "다시 올라가는 행동이 생깁니다.",
    "ddm.residual_in_watch": "다른 콘텐츠를 보는 동안에도 지나간 카드를 생각할지입니다. 켜면 보다가도 \"아까 그거…\"가 떠오릅니다.",
    "ddm.attention": "화면에 아직 뭔지 모르는 카드가 여러 장 있으면 하나하나 알아보기가 느려지는지입니다. "
                     "none = 상관없음, log = 조금 느려짐(3장이면 절반 속도), power = 많이 느려짐(3장이면 1/3 속도). "
                     "켜면 화면이 붐빌 때 멈춰서 보는 시간이 늘어납니다.",
    "ddm.attention_beta": "power를 골랐을 때 붐비는 화면에서 얼마나 더 헷갈리는지입니다. 1이면 여러 장에 주의가 똑같이 나뉩니다.",
    # feed
    "feed.p": "피드에서 정말 볼 만한 콘텐츠가 차지하는 비율입니다. 낮으면 좋은 걸 찾으려고 한참 내려가고, "
              "높으면 금방 찾고 조금씩만 봅니다. (어느 게 좋은 건지는 직접 보기 전엔 모릅니다)",
    "feed.kappa": "좋은 콘텐츠끼리, 별로인 콘텐츠끼리 붙어 있는 정도입니다. 높으면 \"이 근처에 좋은 게 많네\" 하는 구간이 생겨 "
                  "그 근처에 머무르고, 별로인 구간은 빨리 지나가게 됩니다.",
    "feed.a0": "평범한 콘텐츠가 처음 볼 때 주는 재미입니다(숫자는 로그 값이라 직접 바꾸기보다 상황 고르기를 추천).",
    "feed.a1": "좋은 콘텐츠가 평범한 것보다 얼마나 더 재미있는지입니다. 클수록 좋은 걸 찾을 이유가 커집니다.",
    "feed.sigma_A": "같은 종류 안에서도 재미가 콘텐츠마다 얼마나 다른지입니다.",
    "feed.m_nu": "썸네일을 보고 무슨 내용인지 알아보기가 얼마나 쉬운지입니다. 높을수록 금방 알아봅니다.",
    "feed.d_nu": "좋은 콘텐츠의 썸네일이 알아보기 더 어려운지(음수) 쉬운지(양수)입니다. 음수로 하면 "
                 "\"재밌는 건 썸네일만 봐서는 모르겠다\"는 상황이 되어, 오래 멈춰 보거나 좋은 걸 놓칩니다.",
    "feed.sigma_nu": "썸네일 알아보기 난이도가 콘텐츠마다 얼마나 다른지입니다.",
    "feed.m_lambda": "콘텐츠가 얼마나 빨리 질리는지입니다. 높을수록 금방 재미가 떨어져서 짧게 보고 넘기고, "
                     "낮을수록 한 번 고르면 오래 봅니다. (로그 값이라 상황 고르기를 추천)",
    "feed.sigma_lambda": "질리는 속도가 콘텐츠마다 얼마나 다른지입니다. 크면 금방 질리는 것과 오래 볼 만한 것이 섞여 있습니다.",
    # estimate (value-error branch)
    "estimate.sigma_est": "썸네일만 보고 \"재밌겠다/별로겠다\"를 잘못 짐작하는 정도입니다. 0이면 썸네일로 정확히 압니다. "
                          "키우면 기대하고 열었다가 실망해서 바로 닫는 일, 괜찮은 걸 별로라고 생각해 일찍 그만보는 일이 늘어납니다.",
    "estimate.noise_lambda": "\"이건 오래 재밌겠다/금방 질리겠다\"도 착각하는지입니다. 끄면 재미의 크기만 착각합니다.",
    "estimate.tau_learn": "콘텐츠를 열어 보면서 착각을 깨닫기까지 걸리는 시간(초)입니다. 0이면 열자마자 바로 진짜를 압니다. "
                          "길수록 별로인 걸 한동안 계속 보거나, 좋은 걸 잘못 판단해 일찍 닫습니다.",
}

# Controls of the page itself.
UI_HELP: dict[str, str] = {
    "policy": "누가 휴대폰을 조작할지 고릅니다.<br>"
              "• human: 내가 키보드로 직접<br>"
              "• random: 아무렇게나<br>"
              "• scrollfirst: 일정한 속도로 내리다가 괜찮아 보이는 걸 처음 알아보면 바로 열기 (위로는 안 감)<br>"
              "• greedy: 지금까지 알아본 것 중 제일 재밌어 보이는 걸 열기 (지나쳤으면 다시 올라감)<br>"
              "• greedy-dwell: greedy처럼 하되, 아직 뭔지 모르는 카드 앞에서는 잠깐 멈춰서 보기<br>"
              "• ppo:이름: 학습시킨 인공지능",
    "rho": "\"이 정도 재미면 볼 만하다\"는 기준입니다. 알아본 콘텐츠의 지금 재미가 이 값보다 크면 열고, "
           "보다가 재미가 이 값 아래로 떨어지면 닫고 나옵니다. 높이면 까다로워져서 조금만 보고 많이 찾아다니고, "
           "낮추면 아무거나 열고 오래 봅니다. (오른쪽 그래프의 점선)",
    "dwell": "greedy-dwell에서, 아직 뭔지 모르는 카드 앞에서 최대 몇 초까지 멈춰서 볼지입니다. 0이면 멈추지 않고 계속 내립니다.",
    "seed": "어떤 피드(콘텐츠 목록)를 쓸지 고르는 번호입니다. 같은 번호면 같은 피드라서, 설정만 바꿔 공정하게 비교할 수 있습니다.",
    "speed": "재생 속도입니다. 10이 실제 속도, 100이면 10배 빠르게 봅니다.",
    "hidden": "실제로는 열어 보기 전까지 모르는 정보(진짜 재미, 좋은 콘텐츠인지)를 분석용으로 보여 줍니다.",
    "scenario": "자주 궁금해지는 상황을 고르면 관련 설정이 자동으로 바뀝니다. 고른 뒤 아래 '행동 비교'를 누르면 "
                "기본 설정과 비교해 행동이 어떻게 달라지는지 숫자로 볼 수 있습니다.",
    "rates": "지금까지 알아본 콘텐츠들이 지금 열면 얼마나 재밌을지입니다(막대가 높을수록 재밌음). 가로는 목록에서의 위치예요. "
             "파란 배경 = 지금 화면, 파란 테두리 = 화면에 보이는 카드, 빨강 = 보는 중, 주황 윗선 = 이미 본 적 있음, 점선 = 볼 만한 기준.",
    "curve": "보고 있는(없으면 마지막으로 본) 콘텐츠의 재미가 시간이 지나며 줄어드는 모습입니다. "
             "색칠된 넓이가 지금까지 얻은 재미입니다. 곡선이 점선(기준) 아래로 내려가면 보통 닫고 나옵니다.",
    "pause": "늦은 깨달음(지나간 카드를 뒤늦게 알아챔)이나 그 카드로 되돌아와 여는 순간에 자동으로 멈춥니다. "
             "멈춘 뒤 ▶을 누르면 이어서 볼 수 있어요. 뒤늦게 알아챈 카드에는 ★ 표시가 붙습니다.",
    "compare": "지금 설정과 기본 설정으로 같은 피드 여러 개를 빠르게 돌려 보고, 행동이 어떻게 달라지는지 비교합니다. "
               "정책·기준(ρ)·dwell은 둘 다 지금 고른 값을 씁니다.",
}

# Behaviour measures in everyday words (compare panel): key, label, unit, how to read it.
BEHAVIOURS: list[dict[str, str]] = [
    {"key": "return", "label": "얻은 재미 (점수)", "unit": "", "hint": "많을수록 시간을 잘 쓴 것"},
    {"key": "time_watch", "label": "콘텐츠를 본 시간", "unit": "초", "hint": "120초 중 실제로 본 시간"},
    {"key": "time_stay_list", "label": "목록에서 멈춰서 살펴본 시간", "unit": "초", "hint": "스크롤을 멈추고 있던 시간 (목록 끝에 닿아 기다린 시간 포함)"},
    {"key": "max_item_seen", "label": "얼마나 아래까지 내려갔나", "unit": "번째", "hint": "피드를 어디까지 훑었나"},
    {"key": "n_understood", "label": "뭔지 알아본 콘텐츠 수", "unit": "개", "hint": "썸네일을 보고 알아차린 개수"},
    {"key": "n_clicks", "label": "열어 본 횟수", "unit": "번", "hint": ""},
    {"key": "mean_bout", "label": "한 번 열면 보는 시간", "unit": "초", "hint": "평균"},
    {"key": "n_short_bouts", "label": "열자마자 1초 안에 닫은 횟수", "unit": "번", "hint": "기대했다가 실망"},
    {"key": "n_revisits", "label": "봤던 걸 다시 연 횟수", "unit": "번", "hint": ""},
    {"key": "n_late_clicks", "label": "늦게 깨닫고 되돌아가 연 횟수", "unit": "번", "hint": "\"아까 그거…\""},
    {"key": "n_off_screen_understood", "label": "지나간 뒤에 알아차린 수", "unit": "개", "hint": ""},
    {"key": "n_hidden_understood", "label": "다른 걸 보는 동안 알아차린 수", "unit": "개", "hint": ""},
    {"key": "click_log_error", "label": "열 때의 착각 크기", "unit": "", "hint": "0보다 크면 실제보다 좋게 착각"},
    {"key": "n_overestimated_clicks", "label": "실제보다 좋아 보여서 연 횟수", "unit": "번", "hint": "착각"},
    {"key": "mean_unread_on_screen", "label": "화면 속 아직 모르는 카드 수", "unit": "장", "hint": "붐비는 정도"},
]

# Ready-made situations: settings to change, suggested policy, what to look for.
SCENARIOS: list[dict] = [
    {"id": "base", "name": "기본 (교수님 초안 설정)", "set": {}, "expect": "비교의 기준입니다."},
    {"id": "take_break", "name": "30초 뒤 그만 보기 알림 [확장]",
     "set": {"reminder.enabled": True},
     "expect": "30초에 멈춰 계속 볼지 묻습니다. 계속 보면 15초 뒤에 다시 알리고, 그만 보기를 고르면 세션을 끝냅니다. "
               "아래 행동 비교는 알림 응답 없이 끝까지 실행한 정책 결과입니다."},
    {"id": "rare", "name": "좋은 콘텐츠가 드문 피드", "set": {"feed.p": 0.1},
     "expect": "볼 만한 걸 찾으려고 훨씬 아래까지(끝까지) 내려가고, 열어도 금방 닫는 일이 늘어 얻는 재미가 절반 정도로 줄어듭니다."},
    {"id": "common", "name": "좋은 콘텐츠가 흔한 피드", "set": {"feed.p": 0.6},
     "expect": "가까운 곳에서 볼 만한 걸 찾아 한 번에 더 오래 보고, 멀리 내려가지 않습니다. 얻는 재미가 늘어납니다."},
    {"id": "clustered", "name": "좋은 것끼리 몰려 있는 피드", "set": {"feed.kappa": 0.85},
     "expect": "평균은 거의 같지만, 좋은 구간을 만나느냐에 따라 피드마다 결과가 크게 달라집니다. seed를 바꿔 가며 보세요."},
    {"id": "hard_good", "name": "좋은 건 썸네일로 알아보기 어려움", "set": {"feed.d_nu": -0.8},
     "expect": "좋은 콘텐츠를 알아보기 전에 지나쳐 버려서 얻는 재미가 절반 정도로 줄고, 볼 걸 찾아 끝까지 내려갑니다. 정책을 greedy-dwell로 바꿔 멈춰서 보면 나아지는지 확인해 보세요."},
    {"id": "boring_fast", "name": "금방 질리는 콘텐츠", "set": {"feed.m_lambda": round(math.log(1.0), 4)},
     "expect": "한 번 열어도 1초도 안 돼 닫는 일이 3배로 늘고, 더 많이 열어 보며 끝까지 내려갑니다. 얻는 재미는 크게 줄어듭니다."},
    {"id": "long_fun", "name": "오래 볼 만한 콘텐츠", "set": {"feed.m_lambda": round(math.log(0.1), 4)},
     "expect": "한 번 고르면 3배 가까이 오래 보고, 여는 횟수는 절반, 피드는 조금만 내려갑니다."},
    {"id": "fast_scroll", "name": "빠르게 넘기는 사람", "set": {"screen.speeds": [1.5]},
     "expect": "너무 빨리 넘겨서 거의 아무것도 알아보지 못하고 끝까지 가 버립니다(점수가 마이너스). 빠른 스크롤의 대가를 보여 줍니다."},
    {"id": "slow_open", "name": "열고 닫는 데 오래 걸리는 앱", "set": {"time.transition_seconds": 1.0},
     "expect": "열고 닫는 데 시간이 들어 여는 횟수가 줄고 점수가 떨어집니다. (여기 정책들은 \"한 번 열면 더 오래 보기\"로 바꾸지 않습니다. 학습된 인공지능이라면 다를 수 있어요)"},
    {"id": "big_screen", "name": "한 화면에 카드가 많은 앱", "set": {"screen.cards_per_screen": 5.0, "screen.speeds": [0.3]},
     "expect": "같은 속도로 넘겨도 여러 장을 동시에 볼 수 있어, 알아보는 개수와 점수가 조금 늘어납니다."},
    # extension branches (shown only when the setting exists)
    {"id": "misjudge", "name": "썸네일만 보고 착각함 [확장]", "set": {"estimate.sigma_est": 0.6},
     "expect": "기대하고 열었다가 실망해서 바로 닫는 일이 늘어납니다. '열자마자 닫은 횟수'와 '실제보다 좋아 보여서 연 횟수'를 보세요."},
    {"id": "misjudge_slow", "name": "착각을 늦게 깨달음 [확장]", "set": {"estimate.sigma_est": 0.6, "estimate.tau_learn": 3.0},
     "expect": "별로인 걸 한동안 계속 보거나 좋은 걸 일찍 닫아서 얻는 재미가 가장 크게 줄어듭니다."},
    {"id": "late", "name": "아까 그거 괜찮았는데 (늦은 깨달음) [확장]", "set": {"ddm.tau_residual": 2.0},
     "policy": "greedy",
     "expect": "지나간 카드를 뒤늦게 알아차리고 다시 올라가서 여는 행동이 생깁니다. 대신 아래로는 덜 내려갑니다."},
    {"id": "crowded", "name": "화면이 붐비면 헷갈림 [확장]", "set": {"ddm.attention": "power"},
     "policy": "greedy-dwell", "dwell": 0.5,
     "expect": "멈춰서 살펴보는 정책(greedy-dwell)이라 손해가 작습니다. 정책을 greedy(멈추지 않음)로 바꾸면 알아보는 게 크게 줄어 점수가 급락합니다."},
    {"id": "crowded_big", "name": "카드 많은 화면 + 헷갈림 [확장]",
     "set": {"ddm.attention": "log", "screen.cards_per_screen": 5.0, "screen.speeds": [0.3]},
     "expect": "카드가 많아도 한꺼번에 알아보기 어려워서, 큰 화면의 이점이 대부분 사라집니다."},
]
