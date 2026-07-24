# Roadmap: From SAC Baselines to Cooperative Spiders

Goal: go beyond straight-line SAC — build portfolio-grade RL work, open-sourced at every stage, ending with multiple spiders cooperating on a task (mining, mineral search, box-pushing, etc.).

Each step is a shippable milestone: code + GIF + README notes on what worked and what failed.

## Step 1 — Command-Conditioned Locomotion  ⬅️ next up

Train the Ant/spider to follow velocity + heading commands (walk any direction, turn, stop) instead of maximizing forward reward.

- Standard legged-robotics recipe; makes demos interactive (steer with gamepad or waypoints)
- Core skills: reward engineering, observation design
- Deliverable: steerable spider demo

## Step 2 — Massively Parallel Training (MJX or Isaac Lab + PPO)

Move off single-CPU-env SAC to thousands of parallel envs with PPO.

- MJX (MuJoCo on JAX) or Isaac Lab
- The workflow every legged-locomotion lab/company actually uses — biggest resume upgrade
- Deliverable: spider trained at ~4096 parallel envs

## Step 3 — Terrain Curriculum + Domain Randomization

Heightfields, stairs, rocks, slopes — difficulty ramps up as the policy improves.

- Curriculum learning + domain randomization
- Deliverable: spider climbing over rocks (the genuinely impressive video)
- **Started early (2026-07-24):** SpyderDesert-v0 landed — procedural desert
  heightfield (make_terrain.py: fractal dunes + distance-gated mountains,
  which IS a spatial curriculum along +x), terrain-aware health checks, and a
  fine-tune run (`runs/spyder_desert_v0`) seeded from the spyder_walk_v3
  checkpoint. Remaining for this step: per-reset terrain randomization and
  terrain observation (height samples) for non-blind climbing.

## Step 4 — Multi-Spider Cooperation

Start dumb-simple: 2–4 spiders pushing a box too heavy for one, or coverage/foraging with shared reward.

- MAPPO or independent PPO; cooperative MARL is finicky — arrive here with parallel-training infra already solid
- Deliverable: cooperative multi-agent task, open-sourced

## Working Habits

- Open-source each stage as it lands (keep the GIF + README pattern)
- Write a short post per milestone: which reward terms / curriculum choices failed and why — failure analysis is what distinguishes candidates

**Priority if time is short:** Steps 1 + 2 together (a steerable spider trained in MJX at 4096 parallel envs) is the biggest portfolio jump available this month.
