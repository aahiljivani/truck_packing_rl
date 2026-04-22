# MuJoCo Truck Packing

This repository contains a MuJoCo-based truck-packing environment plus a Soft Actor-Critic (SAC) training script for learning box placement policies. The setup is modeled after the Dexterity truck packing challenge: boxes arrive one at a time, the agent chooses a placement and orientation, MuJoCo settles the physics, and the environment returns an observation, reward, and termination signal.

The codebase currently focuses on fast iteration for reward design and training experiments rather than polished packaging. The main workflow is:

1. Generate or reuse a pool of box dimensions.
2. Build a MuJoCo environment that simulates truck packing locally.
3. Train a continuous-control SAC policy over box placements.
4. Inspect reward, density, and episode metrics in TensorBoard or the passive viewer.

## Repository Structure

- `mjc.py`: main Gymnasium environment, MuJoCo model augmentation, reward logic, settling, and diagnostics.
- `sac.py`: SAC training loop adapted from CleanRL for continuous control.
- `sim_details/get_boxes.py`: utility for fetching box sequences and saving them to `sim_details/box_dimensions.json`.
- `sim_details/box_dimensions.json`: cached box dimension pool used for offline sampling.
- `sim_details/sim.xml`: MuJoCo truck geometry used by the environment.
- `sim_details/notes.txt`: local notes and API examples used during development.
- `wrappers.py`, `cleanrl_utils/`: support code used by the RL training loop.

## Environment Summary

The environment class is `MujocoTruckEnv` in `mjc.py`.

### Truck geometry

The truck dimensions are:

- depth: `2.0 m`
- width: `2.6 m`
- height: `2.75 m`

The MuJoCo scene is defined in `sim_details/sim.xml`. Boxes are dynamically injected into the XML at reset time based on the chosen sequence of box dimensions.

### Observation

The observation is a flat float vector containing:

- current box dimensions and weight
- running truck extents
- density
- void ratio
- bird's-eye stability grid
- bird's-eye occupancy/count grid

By default the observation size is `9 + 2 * grid_x * grid_y`, which is `41` when `grid_x = grid_y = 4`.

### Action

The action is a continuous 4D vector:

- `x`
- `y`
- `z`
- `s`

Here `x`, `y`, and `z` are API-frame placement coordinates and `s` is a scalar in `[-1, 1]` that is discretized into one of four box orientations.

### Episode flow

For each step:

1. The agent proposes a placement.
2. The placement is clamped to stay within nominal truck bounds.
3. MuJoCo settles the placed box for a fixed/adaptive number of physics steps.
4. The environment measures density, drift, overlap, void ratio, stability, and other diagnostics.
5. The reward and termination logic are applied.

## Current Reward Design

Reward shaping is being actively iterated. At the moment, the reward in `mjc.py` is centered on two ideas:

- strongly reward increases in density
- penalize placements with larger `x` depth values

The current step reward is approximately:

```python
density_gain = curr_density - prev_density
density_term = alpha * (300 * positive_density_gain + 120 * negative_density_gain)
x_depth_penalty = x_depth_penalty_scale * (x_depth_normalized ** 2)
reward = density_term - x_depth_penalty
reward -= overlap penalties
reward -= settle drift penalty when drift exceeds margin
```

There are also terminal bonuses and penalties:

- unstable placements terminate the episode
- hitting the density threshold terminates with a success-style bonus
- completing all boxes terminates with a final density/void bonus
- out-of-container is tracked, but it is currently configured **not** to terminate episodes during SAC training

Useful info fields emitted by the environment include:

- `density`
- `void`
- `compactness`
- `reward_density_term`
- `reward_x_depth_penalty`
- `x_depth_normalized`
- `settle_drift`
- `drift_penalty`
- `overlap_penalty`
- `termination_reason`

## Training

The main training entry point is:

```bash
python sac.py --num-envs 1 --total-episodes 100 --n-boxes 20 --n-sequences 10
```

On macOS, if you want MuJoCo rendering, use `mjpython`:

```bash
mjpython sac.py --num-envs 1 --render --render-envs 1 --total-episodes 20 --n-boxes 20 --n-sequences 10
```

Short visual smoke test:

```bash
mjpython sac.py --num-envs 1 --render --render-envs 1 --total-episodes 5 --n-boxes 5 --n-sequences 3
```

Larger parallel training run:

```bash
python sac.py \
    --num-envs 8 \
    --total-episodes 1000 \
    --n-boxes 100 \
    --n-sequences 20 \
    --learning-starts 2000 \
    --buffer-size 400000 \
    --total-timesteps 1000000
```

### Important SAC arguments

- `--num-envs`: number of vectorized environments
- `--n-boxes`: boxes per episode
- `--n-sequences`: number of pre-fetched dimension sequences shared across envs
- `--render`: enable passive MuJoCo viewer
- `--render-envs`: how many vector envs should render
- `--learning-starts`: steps before gradient updates begin
- `--buffer-size`: replay buffer size
- `--total-episodes`: stop condition based on completed episodes
- `--total-timesteps`: hard safety cap on environment steps

Training logs are written under `runs/`.

To inspect metrics:

```bash
tensorboard --logdir runs
```

## Box Sequence Generation

If you need a larger or fresher box pool, run:

```bash
python sim_details/get_boxes.py --n_boxes 100
```

This script calls the challenge API, walks through placements to collect box dimensions, and appends them to `sim_details/box_dimensions.json`.

The training script can then sample from that cached pool using `json_box_sequence(...)` in `mjc.py`, which avoids repeatedly calling the API during every environment reset.

## MuJoCo Rendering Notes

- `sac.py` supports headless and rendered training.
- `render_mode="human"` opens the MuJoCo passive viewer.
- On macOS, `mjpython` is the safer entry point for viewer-based runs.
- Rendering many vector envs at once is expensive; start with `--num-envs 1`.

## Development Notes

This project is currently optimized for experimentation:

- reward shaping is changing frequently
- some defaults are intentionally tuned for quick smoke tests
- diagnostics in `info` are richer than the public API state because they are meant to support RL debugging

If training appears stalled, the most important things to inspect are:

- `charts/episodic_return`
- `charts/density`
- termination reasons printed by `sac.py`
- overlap and drift penalties in env `info`
- whether the x-depth penalty is overpowering density gains

## Dependencies

The code expects a Python environment with at least:

- `mujoco`
- `gymnasium`
- `numpy`
- `torch`
- `tyro`
- `tensorboard`
- `requests`

Some older experimentation scripts also import `jax`.

## Quick Start

From the repo root:

```bash
python sim_details/get_boxes.py --n_boxes 100
python sac.py --num-envs 1 --total-episodes 100 --n-boxes 20 --n-sequences 10
tensorboard --logdir runs
```

If you want to watch the policy live on macOS:

```bash
mjpython sac.py --num-envs 1 --render --render-envs 1 --total-episodes 20 --n-boxes 20 --n-sequences 10
```
