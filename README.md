# The Imitator Game

## [[Website](https://imitator-game.github.io/index.html)]  |  [[Document](https://imitator-game.github.io/docs/guide.html)]  |  [[Paper](https://arxiv.org/abs/2608.22301)]  |  [[Dataset](https://imitator-game.github.io/data.html)]  |  [[Checkpoints](https://imitator-game.github.io/data.html#checkpoints)]  |  [[Community](https://imitator-game.github.io/community.html)]

![teaser](docs/media/cover.png)

The Imitator Game is a hierarchical benchmark for robot imitation from human manipulation videos. A human first demonstrates the manipulation skills in front of a camera (the Demonstrator scene); robots then attempt to imitate those skills from video observation alone in the Imitator scene (the Imitator scenes). Evaluation is conducted with blind A/B comparisons in the Imitator Arena or simulation automated metrics to judge which model imitates the human better.

This repo contains the simulation framework (powered by [ManiSkill](https://www.maniskill.ai/) and [SAPIEN](https://sapien.ucsd.edu/)), the IG-10K data collection and preprocessing tooling, and the policy baselines ([ACT](https://arxiv.org/abs/2304.13705), [Diffusion Policy](https://arxiv.org/abs/2303.04137v5), [VQ-BeT](https://arxiv.org/abs/2403.03181), [UniSkill](https://arxiv.org/abs/2505.08787), [XSkill](https://arxiv.org/abs/2307.09955), [GR00T](https://arxiv.org/abs/2503.14734), [RDT](https://arxiv.org/abs/2410.07864), [π0.5](https://arxiv.org/abs/2504.16054), [OpenVLA](https://arxiv.org/abs/2406.09246), [Zero-WAM](https://robbyant-research.github.io/Zero-WAM/)) including their training and evaluation recipes.

## News

- **Oct 28, 2026** – Update benchmark baseline with [Zero-WAM](https://robbyant-research.github.io/Zero-WAM/) code and results.
- **Oct 21, 2026** – Upload baseline [checkpoints](https://huggingface.co/imitator-game/IG-10K-Checkpoints) evaluated in the paper, mirrored on [ModelScope](https://modelscope.cn/models/Zhouxunzhe/IG-10K-Checkpoints).
- **Oct 10, 2026** – 🎉The Imitator Game is accepted by CoRL 2026 and selected as NeurIPS 2026 ODI Workshop **Oral Presentation**.
- **Sep 18, 2026** – Update the [leaderboard](https://imitator-game.github.io/leaderboard.html) with by-task results.
- **Aug 23, 2026** – Release [The Imitator Game paper](https://arxiv.org/abs/2608.22301) and open-source the [code](https://github.com/imitator-game/The-Imitator-Game).

## Installation

The full repository uses [uv](https://docs.astral.sh/uv/getting-started/installation/) and Vulkan for rendering. For the smaller ACT + ManiSkill setup in this reproduction, install only the dependencies listed in `requirements-act.txt`:

```bash
uv venv --python 3.12
uv pip install --no-config --python .venv/bin/python -r requirements-act.txt
# This repository consumes LeRobot 0.5.0 dataset APIs. Install its code without
# resolving its metadata: that release requires Gymnasium >=1.1 and Diffusers <0.36,
# which conflict with ManiSkill's Gymnasium 0.29 and this repo's Diffusers >=0.36.
uv pip install --no-config --no-deps --python .venv/bin/python "lerobot==0.5.0"
```

`--no-config` is needed here because this repository's full-project `uv` overrides
target a different dependency set than the ACT + ManiSkill subset.

Set up Vulkan for rendering following the [ManiSkill installation instructions](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan).

## Downloading assets

We provide the [assets set](https://huggingface.co/datasets/imitator-game/IG-10K-Assets) that we used in The Imitator Game ([ycb](https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip) / [RoboTwin](https://robotwin-platform.github.io/doc/objects/index.html) / [PartNet-Mobility](https://sapien.ucsd.edu/browse) / [sketchfeb](https://sketchfab.com/)), but we encourage the community to add more diverse assets. The assets should be placed in `~/.maniskill/data`.

For the compact CleanDesk + PressJuicer L0 subset, the script below downloads the shared asset archive, the YCB models used by CleanDesk, and only PartNet CoffeeMachine model `103105` from the PartNet archive:

```bash
.venv/bin/python scripts/download_two_task_assets.py
```

The original L0-only selection is downloaded separately with `scripts/download_two_task_l0.py`. For the complete two-task L0–L3 reproduction subset, use the following downloader; it also retrieves the task-specific assets required to evaluate every level:

```bash
.venv/bin/python scripts/download_two_task_l0_l3.py
```

This keeps the shared human `H19`/`H50` demonstrations, all four levels of their real-robot trajectories, and all eight simulation datasets. It downloads ZED2i RGB plus parquet and metadata needed for actions, timing, and MANO features. Existing L0 files are reused. The expected fresh transfer is about 10 GB, including the PartNet archive needed to extract the additional L2 CoffeeMachine model; the retained selected dataset is about 6.9 GiB plus the installed simulator assets. See the [complete subset notes](examples/baselines/lerobot_dataset/config/two_task_l0_l3/README.md) for training and evaluation commands.

Before ACT training, download its frozen DINOv2 task encoder to the local cache:

```bash
.venv/bin/python scripts/download_act_backbone.py
```

For a full benchmark asset download instead, use the commands below.

```bash
# Hugging Face
mkdir -p ~/.maniskill/data
hf download imitator-game/IG-10K-Assets --repo-type dataset --local-dir ~/.maniskill/data
# ModelScope
# modelscope download --dataset Zhouxunzhe/IG-10K-Assets --local_dir ~/.maniskill/data

# Extract
for f in ~/.maniskill/data/*.tar.zst; do tar --use-compress-program=unzstd -xf "$f" -C ~/.maniskill/data; done
rm ~/.maniskill/data/*.tar.zst
```

External GLB assets (Sketchfab, Hunyuan-generated meshes, [Objaverse](https://objaverse.allenai.org/), etc.) use the pipeline documented in [`mani_skill/assets/sketchfab_README.md`](mani_skill/assets/sketchfab_README.md): download the shared assets to `~/.maniskill/data/sketchfab/objects/{object_key}/model.glb`, register them in `mani_skill/assets/sketchfab_registry.json`, and load them in tasks with `create_sketchfab_actor(...)`.

## IG-10K Dataset

The dataset is in `LeRobot-0.5.0` format. See [`examples/baselines/lerobot_dataset/README.md`](examples/baselines/lerobot_dataset/README.md) for the dataset layout, observation features (RGB/depth cameras, MANO hand pose, segmentation masks), and the imitative level handling of the paired human/robot data.

The compact CleanDesk + PressJuicer L0 dataset is downloaded with `scripts/download_two_task_l0.py`. For the complete L0–L3 two-task selection, run `scripts/download_two_task_l0_l3.py`; both scripts use a file allowlist and do not fetch the full dataset.

```bash
# Hugging Face
hf download imitator-game/IG-10K-Dataset --repo-type dataset --local-dir dataset/IG-10K-full
# ModelScope
# modelscope download --dataset Zhouxunzhe/IG-10K-Dataset --local_dir dataset/IG-10K-full
```

The selected CleanDesk + PressJuicer datasets are grouped by source, task, and
level so the corresponding human, real-robot, and simulation data are easy to
see together:

```text
The-Imitator-Game/
└── dataset/
    ├── human_data/
    │   ├── CleanDesk/              # human_H19
    │   └── PressJuicer/            # human_H50
    ├── robot_data/
    │   ├── CleanDesk/L0 … L3/
    │   └── PressJuicer/L0 … L3/
    └── sim_data/
        ├── CleanDesk/L0 … L3/
        └── PressJuicer/L0 … L3/
```

`human_H19` maps to CleanDesk and `human_H50` maps to PressJuicer. The three
sources correspond by task and, for robot/simulation, by L0–L3 level; their
episodes are independent recordings rather than frame-by-frame paired clips.
Human demonstrations are shared across levels. See [dataset/README.md](dataset/README.md)
for the full mapping and file layout.

### Collect new data in simulation

See [`scripts/collect_data.py`](scripts/collect_data.py) for collecting simulation motion-planning demonstrations in ManiSkill hdf5-format, and [examples/baselines/lerobot_dataset/h5_to_lerobot.py](examples/baselines/lerobot_dataset/h5_to_lerobot.py ) for converting `.h5` trajectories to `LeRobot-0.5.0` format. An example script for data collection:

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run --shader "rt-fast" -n 50 -e "Task name" --save-video
```

## Benchmark

Each baseline has its own README.md with training and evaluation commands under [examples/baselines/](examples/baselines/). An example script for model training and evaluation:

```bash
# Training
python -m examples.baselines.model.train_model_imitator \
  --human-root dataset/human_data \
  --sim-root dataset/sim_data \
  --human-dataset-file path/to/human_train_config.json \
  --sim-dataset-file path/to/sim_train_config.json \
  --task-mapping-file examples/baselines/lerobot_dataset/task_mapping.json

# Evaluation
python -m examples.baselines.model.eval_model_imitator \
  --eval-config path/to/eval_config.txt \
  --checkpoint path/to/final_model.pt \
  --human-root dataset/human_data \
  --sim-root dataset/sim_data \
  --human-config path/to/human_eval_config.json \
  --sim-config path/to/sim_eval_config.json \
  --task-mapping examples/baselines/lerobot_dataset/task_mapping.json
```

You can also download the evaluated baseline checkpoints from [Hugging Face](https://huggingface.co/imitator-game/IG-10K-Checkpoints) or [ModelScope](https://modelscope.cn/models/Zhouxunzhe/IG-10K-Checkpoints).

## Community & contributions

The Imitator Game is also an **open-source community** for robot imitation from human
video. Anyone is welcome to
participate and contribute to the community. Contributions will be acknowledged and merged into the project after review.

### **Get involved**

Everyone can contribute to this project by submitting **tasks / models / embodiments** that support the Imitator-Game **data-collection / training / evaluation** framework. Please follow the design templates and contribute via pull request:

- Contribute new Imitator tasks (L0–L3) designed with the task template under [mani_skill/envs/tasks/_template/](mani_skill/envs/tasks/_template/) (together with the motion-planning solutions);
- contribute new imitation-learning models built on the baseline template under [examples/baselines/_template/](examples/baselines/_template/) (together with evaluation interface);
- contribute new robots with robot template under [mani_skill/agents/robots/_template/](mani_skill/agents/robots/_template/).

### Pairing data service

We operate a **data pairing service**: for
community-contributed tasks, we may collect *human and robot pairing demonstrations*, and merge the paired data into IG-10K-format datasets
under the shared LeRobot format. Community members can request pairing for
their accepted tasks.

### Real-world deployment and evaluation

Contributors can submit their policy for deployment on our real-robot
platform. We will run the submitted policy under the agreed evaluation protocols and record the rollout videos. Approved real-world rollouts can be added to the
[Imitator Arena](https://imitator-game.github.io/arena.html), where the community
can make blind A/B comparisons between anonymous model rollouts against the
same human reference. Supported human, real-robot, and simulation tasks can be found in the [Task Gallery](https://imitator-game.github.io/gallery.html). For a real-world evaluation request, please contact us in the community by [WeChat](https://imitator-game.github.io/community.html) or [Discord](https://discord.gg/p8QVXWwWyZ).

## Technical Support

 See the [docs](https://imitator-game.github.io/docs/guide.html) for more details. Contact [Xunzhe Zhou](https://zhouxunzhe.github.io/) or join the [Community](https://imitator-game.github.io/community.html) if you need any support.

## Citation

```bibtex
@misc{zhou2026imitatorgamebenchmarkingrobot,
  title  = {The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction},
  author = {Xunzhe Zhou and Yiyang Cai and Fengyi Wang and Ran Ju and Hanxiang Ren and
            Ruizhe Liu and Yu Zhang and Qian Luo and Feng Chen and Pei Zhou and
            Yi Ma and Yanchao Yang},
  year   = {2026},
  eprint = {2608.22301},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url    = {https://arxiv.org/abs/2608.22301}
}
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
