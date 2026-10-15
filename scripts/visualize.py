"""Watch one episode of a policy.

Run (PowerShell, venv active):
    python scripts\\visualize.py --policy greedy                    # summary figure
    python scripts\\visualize.py --policy greedy --animate          # playback (space = pause)
    python scripts\\visualize.py --policy ppo --run runs\\pilot --animate
    python scripts\\visualize.py --policy ppo --run runs\\pilot --save runs\\pilot\\ep.png
    python scripts\\visualize.py --policy greedy --kappa 0.8 --d-nu -0.7   # another feed
    python scripts\\visualize.py --policy greedy --dwell 1.5 --set screen.cards_per_screen=5

Policies: random, scrollfirst, greedy, ppo (needs --run). --seed picks the list
and DDM noise (10000+ = evaluation lists the PPO never trained on).
--save writes .png (figure) or .gif / .mp4 (with --animate; .mp4 needs ffmpeg).
"""

from __future__ import annotations

import argparse

from scrolling_agent import ScrollingEnv
from scrolling_agent.cli import add_config_args, config_from_args, make_policy
from scrolling_agent.evaluation import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="greedy", choices=["random", "scrollfirst", "greedy", "ppo"])
    parser.add_argument("--run", default=None, help="PPO run folder (runs/<name>)")
    parser.add_argument("--which", default="best", choices=["best", "final"])
    parser.add_argument("--rho", type=float, default=0.3, help="watch threshold of the baselines")
    parser.add_argument("--dwell", type=float, default=0.0, help="greedy: stop up to this long per unread item")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--every", type=int, default=2, help="animation: draw every k-th step")
    parser.add_argument("--interval", type=int, default=50, help="animation: ms per frame")
    parser.add_argument("--hide-hidden", action="store_true", help="animation: do not print hidden A, nu, z")
    parser.add_argument("--save", default=None, help="output file instead of a window")
    add_config_args(parser)
    args = parser.parse_args()

    import matplotlib

    if args.save and not args.animate:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scrolling_agent.viz import animate_episode, episode_summary, plot_episode

    env = ScrollingEnv(config_from_args(args))
    policy = make_policy(args.policy, args.rho, args.run, args.which, dwell=args.dwell)
    _, trace = run_episode(env, policy, seed=args.seed, record=True)
    for k, v in episode_summary(trace).items():
        print(f"{k:<20} {v:.2f}" if isinstance(v, float) else f"{k:<20} {v}")

    title = f"{args.policy}" + (f" ({args.run})" if args.run else "") + f" · seed {args.seed}"
    if not args.animate:
        fig = plot_episode(trace, title)
        if args.save:
            fig.savefig(args.save, dpi=120)
            print(f"Saved {args.save}")
        else:
            plt.show()
        return

    fig, anim = animate_episode(trace, every=args.every, interval=args.interval,
                                show_hidden=not args.hide_hidden)
    fig.suptitle(title)
    if args.save:
        fps = max(1000 // args.interval, 1)
        writer = "ffmpeg" if args.save.endswith(".mp4") else "pillow"
        anim.save(args.save, writer=writer, fps=fps)
        print(f"Saved {args.save}")
        return

    paused = [False]

    def toggle(event):
        if event.key == " ":
            (anim.resume if paused[0] else anim.pause)()
            paused[0] = not paused[0]

    fig.canvas.mpl_connect("key_press_event", toggle)
    plt.show()


if __name__ == "__main__":
    main()
