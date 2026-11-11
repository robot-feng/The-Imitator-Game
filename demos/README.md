# Local reproduction data

The compact CleanDesk + PressJuicer subset is grouped by source so the three
directories match the training and evaluation arguments:

```text
demos/
├── demo_data/       # Human demonstrations
│   ├── human_H19/   # CleanDesk
│   └── human_H50/   # PressJuicer
├── robot_data/      # Real-robot demonstrations
│   ├── robot_H19_L0 … robot_H19_L3
│   └── robot_H50_L0 … robot_H50_L3
└── imitator_data/   # Simulation demonstrations
    ├── L0_TwoRobotCleanDesk-v1 … L3_TwoRobotCleanDesk-v1
    └── L0_TwoRobotPressJuicer-v1 … L3_TwoRobotPressJuicer-v1
```

Human data is shared across levels. Each task directory contains the selected
LeRobot `data/`, `meta/`, and `videos/observation.images.zed2i/` files. The
parquet files contain actions, timestamps, and human MANO features. Other
camera/depth video streams and unrelated tasks are not part of this subset.

Download or resume the complete selection from the repository root:

```bash
.venv/bin/python scripts/download_two_task_l0_l3.py
```

The dataset files are local and ignored by Git; only this layout guide is
tracked. Configs and ACT commands are in
`examples/baselines/lerobot_dataset/config/two_task_l0_l3/README.md`.
