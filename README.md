# MuJoCo Truck Packing

This repository contains a MuJoCo-based truck-packing environment plus a Soft Actor-Critic (SAC) training script adapted from CleanRL for learning box placement policies. The setup is modeled after the Dexterity truck packing challenge: boxes arrive one at a time, the agent chooses a placement and orientation, MuJoCo settles the physics, and the environment returns an observation, reward, and termination signal.

The codebase currently focuses on fast iteration for reward design and training experiments. The main workflow is:

1. Collect a sample of box dimensions and weights from 'dev' mode and save them to a json file.
2. Build a MuJoCo environment that simulates truck packing locally. All observations are normalized to `[0, 1]` and include three bird's-eye grids (stability, count, height).
3. Train a continuous-control SAC policy over box placements.
4. Inspect reward, density, and episode metrics in CLI.

## Repository Structure

- `mjc.py`: main Gymnasium environment, MuJoCo model augmentation, reward logic, settling, and diagnostics.
- `sac.py`: SAC training loop adapted from CleanRL for continuous control.
- `sim_details/get_boxes.py`: utility for fetching box sequences and saving them to `sim_details/box_dimensions.json`.
- `sim_details/box_dimensions.json`: cached box dimension pool used for offline sampling.
- `sim_details/sim.xml`: MuJoCo truck geometry used by the environment.
- `sim_details/notes.txt`: local notes and API examples used during development.
- `wrappers.py`, `cleanrl_utils/`: support code used by the RL training loop.


### Truck geometry

The truck dimensions are:

- depth: `2.0 m`
- width: `2.6 m`
- height: `2.75 m`

The MuJoCo scene is defined in `sim_details/sim.xml`. Boxes are dynamically injected into the XML at reset time based on the chosen sequence of box dimensions.

### Observation

The observation is a single flat `float32` vector with every element normalized to `[0, 1]`. With the default `grid_x = grid_y = 4`, the vector has

```
9 + 3 * grid_x * grid_y = 9 + 48 = 57
```

entries, laid out as:

```
[  l, w, h,                              # 3  current box dimensions (raw, already in [0, 1])
   w_b_norm,                             # 1  current box weight, normalized
   max_x_norm, max_y_norm, max_z_norm,   # 3  running truck extents, normalized
   density, void,                        # 2  geometric packing ratios, both in [0, 1]
   grid_stability (grid_x * grid_y),     # 16 bird's-eye per-cell stability
   grid_count_norm (grid_x * grid_y),    # 16 bird's-eye per-cell normalized count
   grid_height_norm (grid_x * grid_y) ]  # 16 bird's-eye per-cell max top height
```

# Further Details about Box Packing Problem Metrics

#### Density

Density is the fraction of the *used prism* that is occupied by placed-box
volume, matching the Dexterity challenge's scoring definition. Formally:

```
density = sum(l_i * w_i * h_i) / ( max_x_reached * TRUCK["width"] * TRUCK["height"] )
```

where the sum runs over all placed boxes and each `(l_i, w_i, h_i)` is the box's
raw dimensions (orientation-independent, since volume is rotation-invariant).
The denominator uses the *running* reached depth `max_x_reached` rather than
the full truck depth `TRUCK["depth"]`, so a policy that packs tightly into a
small prefix of the truck receives credit before it fills the whole depth.
Before any boxes are placed, or when the denominator is zero, `density = 0.0`.

Density is the primary training signal: the reward term in `mjc.py` rewards
increases in density, and crossing `stop_density` triggers a successful
termination with a terminal bonus.

#### Void ratio

Void ratio measures how much empty space exists *inside the axis-aligned
bounding box of all placed boxes*, in oriented coordinates. It captures "are
the boxes huddled tightly or spread out with gaps" without being fooled by the
truck's empty back half (which is what `density` already handles).

```
for each placed box i:
    l_o, w_o, h_o = IDENTITY_DIMS_MAP[orientation_i](l_i, w_i, h_i)
    half_i = (l_o/2, w_o/2, h_o/2)
    min_i  = position_i - half_i
    max_i  = position_i + half_i
    occupied += l_o * w_o * h_o

bb_min   = min over i of min_i         # componentwise
bb_max   = max over i of max_i
bounding = product(bb_max - bb_min)    # volume of the bounding box

void = clip(1 - occupied / bounding, 0.0, 1.0)
```

`void` is `0.0` when fewer than two boxes are placed (the bounding box collapses)
or when the occupied volume equals the bounding volume (a perfect tiling).
`1.0` means the placed boxes span a large bounding volume while filling almost
none of it.

### Action

The action is a continuous 4D vector:

- `x`
- `y`
- `z`
- `s`

Here `x`, `y`, and `z` are API-frame placement coordinates and `s` is a scalar in `[-1, 1]` that is discretized into one of four box orientations.

### Episode flow

For each step:

1. The agent proposes a placement `(x, y, z, s)`.
2. `s` is discretized into one of four axis-aligned quaternions via
   `IDENTITY_DIMS_MAP`, which also determines the box's oriented extents
   `(l_o, w_o, h_o)`.
3. `(x, y, z)` is clamped per-axis into nominal truck bounds given the oriented
   box size. If some axis interval is empty, the step aborts with an
   `out_of_bounds` termination.
4. MuJoCo settles the placed box for up to `settle_seconds_max` with optional
   adaptive early stopping when linear and angular velocities fall below the
   configured epsilons for `settle_consecutive_steps` consecutive substeps.
5. The environment recomputes density, void ratio, drift, overlap, the three
   bird's-eye grids, and stability diagnostics.
6. The shaped reward is assembled and the termination ladder is checked.

## Current Reward Design

The reward is deliberately minimal. Most of the shaping lives in the
termination ladder rather than the per-step signal, on the premise that
the agent already sees enough state (the stability, count, and height
grids) to learn stability implicitly. The per-step reward should stay
close to the true objective: pack as much volume as possible into as
little depth as possible without toppling.

### Per-step reward

```python
density_gain = curr_density - prev_density
reward = alpha * 100.0 * density_gain
```

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
- `--resume`: path to a checkpoint `.pt` file to resume training from (see
  "Model Checkpoints" below)

Training logs are written under `runs/`. Each run's directory is named

```
runs/<env_id>__<exp_name>__n<n_boxes>__<seed>__<unix_timestamp>/
```

so that curriculum stages with different `n_boxes` are easy to tell
apart at a glance.

To inspect metrics:

```bash
tensorboard --logdir runs
```

## Model Checkpoints

At the end of every training run, `sac.py` writes a single checkpoint
file into the `model/` directory (created on demand). The filename is:

```
model/<exp_name>_n<n_boxes>_<index>.pt
```

where `<index>` auto-increments against any existing files that already
share the `<exp_name>_n<n_boxes>` prefix. So the first run with
`--exp-name sac --n-boxes 15` produces `model/sac_n15_1.pt`, the second
produces `model/sac_n15_2.pt`, and so on. The `n<n_boxes>` piece makes
curriculum stages easy to organize without overwriting each other.

Each checkpoint is a single `torch.save(...)` dict with everything
needed to resume training faithfully:

- `actor`, `qf1`, `qf2`, `qf1_target`, `qf2_target` — network state dicts
- `actor_optimizer`, `q_optimizer` — Adam states for the policy and the
  twin Q-critics
- `log_alpha`, `a_optimizer` — entropy coefficient and its optimizer
  state (when `--autotune` is on, which is the default)
- `global_step`, `episode_count` — for logging continuity
- `args` — the full parsed `Args` dataclass as a dict, for reproducibility

To resume from a checkpoint pass `--resume` with the path:

```bash
python sac.py \
    --num-envs 4 \
    --total-timesteps 120000 \
    --learning-starts 500 \
    --n-boxes 30 \
    --n-sequences 10 \
    --resume model/sac_n15_1.pt
```

On resume, the script skips the uniform-random exploration period
(`--learning-starts` still controls when gradient updates start, but
actions come from the loaded policy instead of random samples) so the
replay buffer refills with on-policy-quality transitions from step 0.

## Curriculum Learning

The environment gets harder non-linearly as `n_boxes` grows: failure
modes compound, stability grids get denser, and the terminal density
ceiling climbs. Training directly on `--n-boxes 200` from a
freshly-initialized policy produces very few `complete` terminations in
reasonable wall-clock time.

The practical workaround is a curriculum: train a policy on a small
`n_boxes`, checkpoint it, then resume into a larger `n_boxes` using
`--resume`. The smaller-task policy already knows how to place boxes
without toppling, so it converges faster on the larger task than a
from-scratch agent does.

A typical three-stage curriculum looks like this:

```bash
# Stage 1: 15-box warm-up. Lots of complete terminations,
# agent learns the basic "don't topple, fill the back wall" behavior.
python sac.py \
    --num-envs 4 \
    --total-timesteps 80000 \
    --learning-starts 2000 \
    --n-boxes 15 \
    --n-sequences 10
# -> model/sac_n15_1.pt

# Stage 2: 30-box refinement. Resume from stage 1.
# Use a smaller --learning-starts since the policy is already
# useful and we want gradient updates flowing almost immediately.
python sac.py \
    --num-envs 4 \
    --total-timesteps 120000 \
    --learning-starts 500 \
    --n-boxes 30 \
    --n-sequences 10 \
    --resume model/sac_n15_1.pt
# -> model/sac_n30_1.pt

# Stage 3: 100-box target. Resume from stage 2.
python sac.py \
    --num-envs 4 \
    --total-timesteps 250000 \
    --learning-starts 500 \
    --n-boxes 100 \
    --n-sequences 20 \
    --resume model/sac_n30_1.pt
# -> model/sac_n100_1.pt
```

Rough heuristics for when a stage is "done" and you can graduate to
the next one:

- the `complete` termination rate is reliably above ~20%, and
- the mean density on `complete` episodes is climbing into the
  0.4+ range, and
- the max density on `unstable` episodes is close to the density on
  recent `complete` episodes (the agent is "almost getting it right"
  even when it fails).

Note that the critic and entropy coefficient reset their target
distributions when `n_boxes` changes, because reward magnitudes scale
with episode length. Give the resumed run a few thousand steps before
evaluating it — early `qf_loss` and `alpha` can look noisy until the
value head recalibrates to the new horizon.

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

## Dependencies

All runtime dependencies are pinned in [`requirements.txt`](requirements.txt). The
recommended install flow inside a fresh conda environment is:

```bash
conda create -n truck_packing python=3.13 -y
conda activate truck_packing
pip install -r requirements.txt
```


## Quick Start

```bash
python sim_details/get_boxes.py --n_boxes 100
python sac.py --num-envs 1 --total-timesteps 20000 --n-boxes 15 --n-sequences 10
tensorboard --logdir runs
```

This trains a small curriculum-stage-1 policy on 15-box episodes and
saves `model/sac_n15_1.pt` when it finishes. To continue from that
checkpoint into a harder task, re-launch with `--resume`:

```bash
python sac.py \
    --num-envs 4 \
    --total-timesteps 120000 \
    --learning-starts 500 \
    --n-boxes 30 \
    --n-sequences 10 \
    --resume model/sac_n15_1.pt
```
