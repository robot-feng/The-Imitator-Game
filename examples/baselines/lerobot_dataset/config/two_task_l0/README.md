# CleanDesk + PressJuicer, L0 subset

This subset selects the 50 training episodes for the following paired tasks:

- `human_H19` ↔ `L0_TwoRobotCleanDesk-v1` ↔ `robot_H19_L0`
- `human_H50` ↔ `L0_TwoRobotPressJuicer-v1` ↔ `robot_H50_L0`

Each task has three related branches: human demonstrations (including video and
MANO estimates), real-robot trajectories, and simulation trajectories. The
selected pairings are `human_H19` → CleanDesk and `human_H50` → PressJuicer.
`human_train.json` is shared by the ACT data modes. Use `sim_train.json` with
`--data-source sim` to pair human videos with simulation trajectories; use
`robot_train.json` with `--data-source robot` to pair the same human videos with
real-robot trajectories. These train separately. `eval_envs.txt` lists the two
L0 simulation environments for policy evaluation.

The ACT video-only configuration defaults to the `zed2i` RGB stream and does not
use depth or segmentation masks. The full LeRobot metadata and parquet files are
still required for episode indexing, actions, timing, and the saved MANO fields.
MANO is part of the human parquet data; it does not need a separate download.

## Download

From the repository root, after installing the environment:

```bash
.venv/bin/python scripts/download_two_task_l0.py
```

This downloads only the two tasks' 50 L0 training episodes for each branch,
human and robot `zed2i` RGB videos, and the associated parquet and metadata
files and places them directly under `demos/demo_data` (human),
`demos/robot_data` (real robot), and `demos/imitator_data` (simulation). It
skips other cameras, depth videos, masks, tasks, and levels.

The selected dataset files total about 2.44 GB. Evaluation also needs the
simulator assets: the shared asset archive (about 27 MB), YCB models
`026_sponge` and `061_foam_brick` for CleanDesk, and PartNet-Mobility model
`103105` for PressJuicer. The PartNet download is a 5.61 GB compressed archive;
only model `103105` needs to be extracted for this subset. Together, the dataset
and required asset downloads are about 8.1 GB, before small metadata files.

## Training modes

Simulation-paired ACT training uses:

```text
--data-source sim
--human-dataset-file examples/baselines/lerobot_dataset/config/two_task_l0/human_train.json
--sim-dataset-file examples/baselines/lerobot_dataset/config/two_task_l0/sim_train.json
```

Real-robot-paired ACT training uses:

```text
--data-source robot
--robot-root demos/robot_data
--human-dataset-file examples/baselines/lerobot_dataset/config/two_task_l0/human_train.json
--robot-dataset-file examples/baselines/lerobot_dataset/config/two_task_l0/robot_train.json
```

The simulation environment evaluation list is in `eval_envs.txt`.

## ACT run on the available GPU

Download the frozen task-video encoder once. The script saves DINOv2 Base to
the local cache and verifies each file before training:

```bash
.venv/bin/python scripts/download_act_backbone.py
```

On the 10 GiB RTX 2080 Ti, the reference ACT transformer settings and batch
size do not fit comfortably. This command keeps the dual-arm ACT policy,
pretrained ResNet18 image backbone, DINOv2 Base task encoder, 24-step action
chunks, both selected task pairings, and the reference 10-epoch schedule. It
uses a 512-wide, 6-layer transformer and batch size 16 to fit this GPU. This is
a hardware-sized reproduction run, not the paper's 1024-wide, 12-layer model.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 .venv/bin/python -m examples.baselines.act.train_act_imitator \
  --exp-name two_task_l0_act_10ep_512 \
  --data-source sim \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-dataset-file examples/baselines/lerobot_dataset/config/two_task_l0/human_train.json \
  --sim-dataset-file examples/baselines/lerobot_dataset/config/two_task_l0/sim_train.json \
  --task-mapping-file examples/baselines/lerobot_dataset/task_mapping.json \
  --frozen-backbone-type dinov2_vitb14 \
  --frozen-backbone-model "$HOME/.cache/imitator-game/models/facebook/dinov2-base" \
  --video-backend pyav \
  --batch-size 16 \
  --num-dataload-workers 0 \
  --task-num-frames 10 \
  --frozen-backbone-num-frames 10 \
  --total-epochs 10 \
  --warmup-epochs 1 \
  --hidden-dim 512 \
  --dim-feedforward 2048 \
  --nheads 8 \
  --enc-layers 6 \
  --dec-layers 6 \
  --save-epoch-freq 5 \
  --no-eval
```

The simulation environments can then be evaluated with 10 episodes each and
the default 500-step limit:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 .venv/bin/python -m examples.baselines.act.eval_act_imitator \
  --eval-config examples/baselines/lerobot_dataset/config/two_task_l0/eval_envs.txt \
  --checkpoint runs/two_task_l0_act_10ep_512-20261110/checkpoints/final_model.pt \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-config examples/baselines/lerobot_dataset/config/two_task_l0/human_train.json \
  --sim-config examples/baselines/lerobot_dataset/config/two_task_l0/sim_train.json \
  --task-mapping examples/baselines/lerobot_dataset/task_mapping.json \
  --input-mode video_only \
  --num-episodes 10 \
  --num-envs 1 \
  --max-episode-steps 500 \
  --control-mode pd_joint_pos \
  --obs-mode rgb \
  --sim-backend physx_cpu \
  --output-dir runs/two_task_l0_act_10ep_512-20261110/eval
```

Use `--data-source robot --robot-root demos/robot_data --robot-dataset-file
examples/baselines/lerobot_dataset/config/two_task_l0/robot_train.json` in place
of the simulation data options to train the same two human-conditioned tasks
against real-robot trajectories. Human demonstrations supply the video task
condition in both modes.

## Completed run (2026-11-10)

The two-task simulation run used the settings above for 10 epochs (19,050
optimizer updates) and finished with mean training loss 0.0400. The final
checkpoint is `runs/two_task_l0_act_10ep_512-20261110/checkpoints/final_model.pt`
(75.11M parameters). Evaluation used 10 episodes per environment, up to 500
steps per episode:

| Environment | Success once | Success at end | Mean return | wTSS |
|---|---:|---:|---:|---:|
| `L0_TwoRobotCleanDesk-v1` | 1.00 | 1.00 | 412.97 ± 2.58 | 0.6003 |
| `L0_TwoRobotPressJuicer-v1` | 0.00 | 0.00 | 55.67 ± 53.37 | 0.0000 |

The rollout JSON, CSV, and videos are in
`runs/two_task_l0_act_10ep_512-20261110/eval/`. This first evaluation used the
then-default 4-step replanning window. The PressJuicer rollout stayed near its
initial pose. A follow-up action trace showed that the checkpoint predicts
24-step chunks; refreshing every 4 steps kept replaying the stationary start of
the PressJuicer motion. The evaluator now defaults to executing the checkpoint's
full predicted chunk before replanning.

## Evaluation follow-up (2026-11-11)

The full 24-step action window was evaluated for 10 episodes per environment.
This confirms that the no-motion behavior came from the short replanning window.
PressJuicer now moves and succeeds in 3/10 episodes, so policy performance on
that task still needs improvement.

| Environment | Success once | Success at end | Mean return | wTSS |
|---|---:|---:|---:|---:|
| `L0_TwoRobotCleanDesk-v1` | 1.00 | 1.00 | 416.06 ± 1.66 | 0.6004 |
| `L0_TwoRobotPressJuicer-v1` | 0.30 | 0.30 | 173.98 ± 180.24 | 0.1812 |

The corrected evaluation outputs are in
`runs/two_task_l0_act_10ep_512-20261110/eval_chunk24/`. A PressJuicer-only
fine-tuning run is a reasonable next experiment; the current 10-epoch,
hardware-sized two-task run does not match the paper's reported benchmark
performance.
