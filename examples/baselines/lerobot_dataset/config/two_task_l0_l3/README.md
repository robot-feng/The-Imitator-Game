# CleanDesk + PressJuicer, complete L0–L3 subset

This reproduction selection contains two tasks at every imitative level:

| Task | Human branch | Real-robot branches | Simulation environments |
|---|---|---|---|
| CleanDesk | `human_H19` | `robot_H19_L0` … `robot_H19_L3` | `L0` … `L3_TwoRobotCleanDesk-v1` |
| PressJuicer | `human_H50` | `robot_H50_L0` … `robot_H50_L3` | `L0` … `L3_TwoRobotPressJuicer-v1` |

The human branch is shared across levels. Each human, real-robot, and simulation
branch contributes its first 50 episodes. The selected streams include ZED2i
RGB videos, LeRobot parquet files, metadata, and MANO fields in the human
parquet data. Other camera streams, depth videos, masks, unrelated tasks, and
unselected episodes are omitted. The training configs use all 50 selected
episodes; evaluation is performed in the eight simulator environments listed
in `eval_envs.txt`.

Run the resumable downloader from the repository root:

```bash
.venv/bin/python scripts/download_two_task_l0_l3.py
```

It reuses valid files already present in `demos/`, stores the data directly in
`demo_data` (human), `robot_data` (real robot), and `imitator_data`
(simulation), writes the configs in this directory, and installs the selected
L2/L3 simulation assets. The expected new
transfer is about 10 GiB when starting from the existing L0 subset; the final
selected dataset is about 6.9 GiB, plus installed assets. The script prints the
exact manifest size before downloading and checks all required assets at the
end. To fetch only a subset of levels, pass for example `--levels 0 1`; the
default is all of L0–L3.

## Simulation-paired ACT training

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 .venv/bin/python -m examples.baselines.act.train_act_imitator \
  --exp-name two_task_l0_l3_act_sim \
  --data-source sim \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-dataset-file examples/baselines/lerobot_dataset/config/two_task_l0_l3/human_train.json \
  --sim-dataset-file examples/baselines/lerobot_dataset/config/two_task_l0_l3/sim_train.json \
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

For training with real-robot trajectories, use the same human config and task
mapping, set `--data-source robot --robot-root demos/robot_data`, and replace
`--sim-dataset-file .../sim_train.json` with
`--robot-dataset-file .../robot_train.json`.

## Simulation evaluation across all levels

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 .venv/bin/python -m examples.baselines.act.eval_act_imitator \
  --eval-config examples/baselines/lerobot_dataset/config/two_task_l0_l3/eval_envs.txt \
  --checkpoint runs/two_task_l0_l3_act_sim-<run>/checkpoints/final_model.pt \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-config examples/baselines/lerobot_dataset/config/two_task_l0_l3/human_train.json \
  --sim-config examples/baselines/lerobot_dataset/config/two_task_l0_l3/sim_train.json \
  --task-mapping examples/baselines/lerobot_dataset/task_mapping.json \
  --input-mode video_only \
  --num-episodes 10 \
  --num-envs 1 \
  --max-episode-steps 500 \
  --control-mode pd_joint_pos \
  --obs-mode rgb \
  --sim-backend physx_cpu \
  --output-dir runs/two_task_l0_l3_act_sim-<run>/eval
```

The `<run>` placeholder should match the timestamped run directory created by
training. Evaluation covers both tasks at L0, L1, L2, and L3. On a 10 GiB GPU,
the 512-wide model and batch size 16 above follow the hardware-sized ACT setup
used for the L0 reproduction; they do not claim the paper's full model size.
