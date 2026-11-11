# CleanDesk + PressJuicer dataset

The selected IG-10K data is grouped first by source, then by task and level:

```text
dataset/
├── human_data/
│   ├── CleanDesk/                 # human_H19
│   └── PressJuicer/               # human_H50
├── robot_data/
│   ├── CleanDesk/L0 … L3/          # robot_H19_L0 … robot_H19_L3
│   └── PressJuicer/L0 … L3/        # robot_H50_L0 … robot_H50_L3
└── sim_data/
    ├── CleanDesk/L0 … L3/          # L0 … L3_TwoRobotCleanDesk-v1
    └── PressJuicer/L0 … L3/        # L0 … L3_TwoRobotPressJuicer-v1
```

## How the three sources correspond

| Task | Human | Real robot | Simulation |
|---|---|---|---|
| CleanDesk | `human_H19` | `robot_H19_L0` … `robot_H19_L3` | `L0_TwoRobotCleanDesk-v1` … `L3_TwoRobotCleanDesk-v1` |
| PressJuicer | `human_H50` | `robot_H50_L0` … `robot_H50_L3` | `L0_TwoRobotPressJuicer-v1` … `L3_TwoRobotPressJuicer-v1` |

These are task-level pairings. The human, real-robot, and simulation episodes
were collected independently, so an episode in one source is not a frame-by-
frame match to an episode in another source. Human demonstrations are shared
across L0–L3; robot and simulation branches each have 50 selected episodes per
task and level. The mapping used by training is recorded in
`examples/baselines/lerobot_dataset/task_mapping.json`.

The names inside `meta/tasks.parquet` are not uniform across the source
datasets: human rows use IDs such as `human_H19`, robot rows can say only
`robot`, and simulation rows use the environment ID. That is why pairing is
defined by the dataset/repository IDs in `task_mapping.json`, rather than by
the task string stored inside each parquet file. `H19` and `H50` identify the
human task datasets; they are not level numbers.

Each dataset directory contains LeRobot `data/`, `meta/`, and the selected
`videos/observation.images.zed2i/` stream. Human parquet files also contain
MANO hand features. Other camera/depth videos and unrelated tasks are omitted.

## Download or resume

Run from the repository root:

```bash
.venv/bin/python scripts/download_two_task_l0_l3.py
```

The downloader preserves this source/task/level layout, reuses valid files,
and writes the matching training and evaluation configs under
`examples/baselines/lerobot_dataset/config/two_task_l0_l3/`. Dataset files are
local and ignored by Git; this README and the configs are tracked.

Training and evaluation commands are in
`examples/baselines/lerobot_dataset/config/two_task_l0_l3/README.md`.
