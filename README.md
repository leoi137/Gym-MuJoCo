# Ant-v5 with SAC (Stable-Baselines3 + MuJoCo)

> 🚧 **Work in progress.** Functional and reproducible today; actively being polished. Expect frequent updates — issues and PRs welcome.

Train a Soft Actor-Critic agent on the Gymnasium `Ant-v5` environment on GPU. Each experiment lives in its own directory under `runs/<run-name>/`, so different reward shapings, seeds, and hyperparameter sweeps can coexist without clobbering each other.

## The two trained policies

<p align="center">
  <img src="assets/baseline_2leg.gif" alt="Baseline SAC policy on Ant-v5" width="400"/>
  &nbsp;&nbsp;
  <img src="assets/foot_contact_v1.gif" alt="Foot-contact-shaped SAC policy on Ant-v5" width="400"/>
</p>

<p align="center">
  <em><strong>Left:</strong> baseline (default Ant-v5 reward) — converged to a two-legged gait.</em>
  &nbsp;&nbsp;
  <em><strong>Right:</strong> foot-contact reward shaping — uses all four legs.</em>
</p>

| Run | Reward function | Steps | Best eval return | Gait |
| --- | --- | --- | --- | --- |
| `baseline_2leg` | default Ant-v5 | 3.75M | 6,657 (unshaped) | two legs only |
| `foot_contact_v1` | default + foot-contact penalty | 3.75M | 5,647 (shaped) | uses all four |

The baseline scores higher in raw forward-velocity reward because it doesn't pay the shaping penalty, but it converged to a degenerate gait. The foot-contact run intentionally trades a bit of forward velocity for a four-legged gait that actually looks like quadrupedal locomotion. See [Why two trained policies?](#why-two-trained-policies) for the full story.

## Install

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` pins PyTorch 2.5.1 built against CUDA 12.1. For CPU-only or a different CUDA version, drop the `--extra-index-url` line and follow <https://pytorch.org/get-started/locally/>.

## Stack

- Python 3.13 (venv in `./venv`)
- PyTorch 2.5.1 + CUDA 12.1
- Gymnasium 1.2 with MuJoCo 3.8
- Stable-Baselines3 2.8

GPU: NVIDIA GeForce RTX 2080.

Activate the venv in every new terminal:

```bash
source venv/bin/activate
```

## Quick start

Watch either trained policy in a live MuJoCo window:

```bash
python watch.py --run baseline_2leg        # the two-legged baseline
python watch.py --run foot_contact_v1      # the four-legged shaped policy
```

Start a fresh experiment of your own:

```bash
# Baseline reward, default hyperparameters, 1M steps
python train.py --run-name my_baseline --seed 0 --steps 1_000_000

# Foot-contact reward shaping
python train.py --run-name my_shaped --seed 0 --steps 1_000_000 \
                --wrapper foot_contact \
                --wrapper-kwargs '{"penalty": 1.0, "window": 50, "contact_threshold": 1.0}'
```

Resume an interrupted run with the same `--run-name`:

```bash
python train.py --run-name my_shaped --steps 2_000_000
# wrapper / seed are read from runs/my_shaped/config.json automatically
```

## Run directory layout

Each invocation of `train.py` writes everything under `runs/<run-name>/`:

```
runs/foot_contact_v1/
├── ant_sac.zip          # latest checkpoint — used to resume
├── ant_sac_best.zip     # best-ever eval policy — used by watch.py
├── ant_sac_best.txt     # best-eval reward (resume-safe high-water mark)
├── ant_buffer.pkl       # saved replay buffer (~1.7 GB)
├── ant_tb/              # TensorBoard event files
├── videos/              # one MP4 per eval snapshot
└── config.json          # wrapper, kwargs, seed, hparams that produced this run
```

`config.json` is the source of truth for what produced a run. On resume, it overrides whatever `--wrapper` or `--seed` you pass on the CLI — this is intentional, so you can't accidentally change reward semantics mid-run and contaminate the replay buffer.

## Two checkpoints per run

RL policies can briefly degrade late in training (catastrophic forgetting / temporary regression). Each run keeps two:

- **`ant_sac.zip`** is always overwritten with the latest model — this is what `train.py` reads to resume.
- **`ant_sac_best.zip`** is only overwritten when an eval beats the previous best — this is what `watch.py` loads by default. The high-water mark survives across runs via `ant_sac_best.txt`.

A resumed run that goes worse won't lose you anything: you can still watch your best-ever policy and keep training from the most recent state. Use `python watch.py --run <name> --latest` to override and watch the latest checkpoint instead of the best.

## Watch progress over time (videos)

Every `--video-every` env-steps (default 50,000), `train.py` rolls one greedy eval episode and writes an MP4 to `runs/<run-name>/videos/`, named by global step:

```
runs/foot_contact_v1/videos/eval_step_000050000.mp4
runs/foot_contact_v1/videos/eval_step_000100000.mp4
...
runs/foot_contact_v1/videos/eval_step_003750000.mp4
```

Open the folder and play them in order to literally see the ant evolve from random flailing into a smooth gait.

## TensorBoard

In a second terminal, point TensorBoard at all runs to compare them side-by-side:

```bash
source venv/bin/activate
tensorboard --logdir runs/
```

Open <http://localhost:6006>. Key metrics:

| Tag                       | What it means                                          |
| ------------------------- | ------------------------------------------------------ |
| `rollout/ep_rew_mean`     | average episode return — the headline training curve   |
| `rollout/ep_len_mean`     | episode length; rises to 1000 as the ant stops falling |
| `eval/mean_reward`        | shaped return on the deterministic eval episode        |
| `eval/base_reward`        | **unshaped** Ant-v5 reward — for apples-to-apples comparison across runs |
| `eval/mean_idle_legs`     | avg legs with no recent ground contact (lower = better quadrupedal gait) |
| `eval/best_mean_reward`   | high-water mark for the run                            |
| `train/actor_loss`        | SAC actor loss                                         |
| `train/critic_loss`       | SAC critic (Q) loss                                    |
| `train/ent_coef`          | auto-tuned entropy temperature α                       |

`eval/base_reward` and `eval/mean_idle_legs` only have data for runs that used a wrapper — the baseline didn't log them because the wrapper didn't exist yet.

## Why two trained policies?

The baseline run (`baseline_2leg`) optimized the stock Ant-v5 reward: forward velocity, minus control cost, minus contact cost, plus a survival bonus. None of those terms encode "use all four legs" — they only encode "move forward without falling". SAC duly found a two-legged hopping gait that maximizes that reward function. It works, but it doesn't look like a quadruped.

The shaped run (`foot_contact_v1`) adds one extra term: for each step, count how many ankles have made ground contact in the last 50 steps, and penalize the agent for each leg that hasn't. The penalty is small (1.0 per idle leg per step) but consistent, so policies that drag two legs are strictly worse than policies that use all four. The forward-velocity term still does the heavy lifting; the wrapper just removes one bad local optimum from the optimization landscape.

The wrapper lives in `wrappers.py` as `FootContactRewardWrapper` and is registered in the `WRAPPERS` dict so any new reward-shaping idea can be added in one place.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | train / resume SAC on Ant-v5 (per-run output dir, optional wrapper) |
| `watch.py` | render a chosen run's best policy in a window |
| `wrappers.py` | reward-shaping wrappers (currently `FootContactRewardWrapper`) and the `WRAPPERS` registry |
| `runs/<name>/` | one self-contained experiment — model, buffer, TB logs, videos, config |
| `assets/` | GIFs used by this README |

## What to expect (SAC on Ant-v5, default reward)

| Steps      | Behavior                                                  |
| ---------- | --------------------------------------------------------- |
| 0 – 50k    | Random flailing, falls over constantly. Returns near 0.   |
| 50k – 150k | Learns to stand, then awkward shuffling. Returns 500–1500.|
| 150k – 300k| A recognizable gait emerges. Returns 2000–3500.           |
| 300k – 500k| Smoother gait. Returns 3500–5500.                         |
| 1M+        | "Solved" territory (~6000+).                              |

With foot-contact shaping, the curves track the same shape but base reward grows a bit slower — the policy is forced to explore four-legged gaits before locking in a strategy.
