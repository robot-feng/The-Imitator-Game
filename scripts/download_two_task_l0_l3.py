"""Download the complete CleanDesk + PressJuicer L0-L3 subset.

The dataset selection includes 50 episodes per human, real-robot, and simulation
branch. Human demonstrations are shared across levels. Only the task assets
needed to evaluate these two tasks are installed.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import subprocess
import time
from zipfile import ZipFile

import requests

from hf_download_utils import download_file, list_files


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "demos"
CONFIG_ROOT = ROOT / "examples/baselines/lerobot_dataset/config/two_task_l0_l3"
ASSET_ROOT = Path(os.getenv("MS_ASSET_DIR", Path.home() / ".maniskill")) / "data"
DATA_REPO = "imitator-game/IG-10K-Dataset"
ASSET_REPO = "imitator-game/IG-10K-Assets"
DATASET_LOCAL_ROOTS = {
    "imitator_human_v1/": "demo_data/",
    "imitator_robot_v1/": "robot_data/",
    "imitator_sim_v1_zed2i/": "imitator_data/",
}
ROBOTWIN_REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
ROBOTWIN_OBJECTS_URL = (
    "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/"
    f"{ROBOTWIN_REVISION}/objects.zip"
)

TASKS = {
    "CleanDesk": {"human": "human_H19", "robot": "robot_H19"},
    "PressJuicer": {"human": "human_H50", "robot": "robot_H50"},
}
ROBOTWIN_MODELS = {
    "023_tissue-box": 3,
    "086_woodenblock": 1,
    "004_fluted-block": 1,
    "002_bowl": 2,
    "035_apple": 0,
}
YCB_MODELS = ("026_sponge", "036_wood_block", "061_foam_brick")
PARTNET_MODEL = "103134"


def _local_dataset_path(remote_path: str) -> Path:
    for remote_root, local_root in DATASET_LOCAL_ROOTS.items():
        if remote_path.startswith(remote_root):
            return DATA_ROOT / local_root / remote_path[len(remote_root) :]
    raise ValueError(f"Unrecognized IG-10K dataset path: {remote_path}")


def _episode_config(repos: list[str]) -> list[dict[str, str]]:
    return [
        {"repo_id": repo, "root": repo, "train": "0:50", "test": ""}
        for repo in repos
    ]


def _write_configs(levels: list[int]) -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    human_repos = [TASKS[task]["human"] for task in TASKS]
    robot_repos = [
        f"{TASKS[task]['robot']}_L{level}"
        for task in TASKS
        for level in levels
    ]
    sim_repos = [
        f"L{level}_TwoRobot{task}-v1"
        for task in TASKS
        for level in levels
    ]
    for name, repos in (
        ("human_train.json", human_repos),
        ("robot_train.json", robot_repos),
        ("sim_train.json", sim_repos),
    ):
        (CONFIG_ROOT / name).write_text(
            json.dumps(_episode_config(repos), indent=2) + "\n",
            encoding="utf-8",
        )
    (CONFIG_ROOT / "eval_envs.txt").write_text(
        "".join(f"L{level}_TwoRobot{task}-v1\n" for level in levels for task in TASKS),
        encoding="utf-8",
    )


def _dataset_branches(levels: list[int]) -> tuple[list[str], list[str], list[str]]:
    humans = [f"imitator_human_v1/{TASKS[task]['human']}" for task in TASKS]
    robots = [
        f"imitator_robot_v1/{TASKS[task]['robot']}_L{level}"
        for task in TASKS
        for level in levels
    ]
    simulations = [
        f"imitator_sim_v1_zed2i/L{level}_TwoRobot{task}-v1"
        for task in TASKS
        for level in levels
    ]
    return humans, robots, simulations


def _list_dataset_files(levels: list[int]) -> dict[str, dict]:
    humans, robots, simulations = _dataset_branches(levels)
    branches = humans + robots + simulations
    folders = [
        f"{branch}/{subpath}"
        for branch in branches
        for subpath in ("data", "meta", "videos/observation.images.zed2i")
    ]
    simulation_roots = set(simulations)
    selected: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(list_files, DATA_REPO, "dataset", folder): folder
            for folder in folders + simulations
        }
        for future in as_completed(futures):
            folder = futures[future]
            for item in future.result():
                if folder in simulation_roots and item["path"] != f"{folder}/.complete":
                    continue
                selected[item["path"]] = item
    return selected


def _download_dataset(levels: list[int]) -> tuple[int, int]:
    selected = _list_dataset_files(levels)
    total = sum(int(item["size"]) for item in selected.values())
    already = sum(
        int(item["size"])
        for item in selected.values()
        if _local_dataset_path(item["path"]).is_file()
        and _local_dataset_path(item["path"]).stat().st_size == int(item["size"])
    )
    print(
        f"Dataset files: {len(selected)}, total {total / 1024**3:.2f} GiB; "
        f"already present {already / 1024**3:.2f} GiB; "
        f"to fetch {max(0, total - already) / 1024**3:.2f} GiB.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(
                download_file,
                DATA_REPO,
                "dataset",
                item,
                _local_dataset_path(item["path"]),
            ): item
            for item in selected.values()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            future.result()
            print(f"[{index}/{len(selected)}] {item['path']}", flush=True)
    return total, already


class _HttpRangeFile:
    """Seekable file-like reader that fetches aligned byte ranges over HTTP."""

    BLOCK_SIZE = 8 * 1024 * 1024

    def __init__(self, url: str):
        response = requests.head(url, allow_redirects=True, timeout=(15, 60))
        response.raise_for_status()
        self.url = response.url
        self.size = int(response.headers["content-length"])
        if "bytes" not in response.headers.get("accept-ranges", "").lower():
            raise RuntimeError("RoboTwin object archive does not support HTTP range requests")
        self.position = 0
        self._cache_start = -1
        self._cache = b""
        self._session = requests.Session()

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        else:
            raise ValueError(f"Invalid seek mode: {whence}")
        if not 0 <= self.position <= self.size:
            raise ValueError(f"Seek outside archive: {self.position}")
        return self.position

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.size - self.position
            if length > self.BLOCK_SIZE:
                raise RuntimeError("Refusing an unbounded read from the remote asset archive")
        remaining = min(length, self.size - self.position)
        chunks = []
        while remaining:
            block_start = (self.position // self.BLOCK_SIZE) * self.BLOCK_SIZE
            if block_start != self._cache_start:
                block_end = min(block_start + self.BLOCK_SIZE, self.size) - 1
                last_error = None
                for attempt in range(5):
                    try:
                        response = self._session.get(
                            self.url,
                            headers={"Range": f"bytes={block_start}-{block_end}"},
                            timeout=(20, 120),
                        )
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise RuntimeError("Remote server ignored a byte-range request")
                        self._cache = response.content
                        self._cache_start = block_start
                        break
                    except Exception as error:
                        last_error = error
                        if attempt < 4:
                            time.sleep(min(2**attempt, 8))
                else:
                    raise RuntimeError("Unable to read the RoboTwin asset archive") from last_error
            within = self.position - self._cache_start
            take = min(remaining, len(self._cache) - within)
            if take <= 0:
                raise IOError("Invalid range returned by the RoboTwin asset server")
            chunks.append(self._cache[within : within + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


def _download_robotwin_models() -> None:
    destination_root = ASSET_ROOT / "robotwin"
    required = []
    for model_name, model_id in ROBOTWIN_MODELS.items():
        required.extend(
            [
                f"objects/{model_name}/model_data{model_id}.json",
                f"objects/{model_name}/collision/base{model_id}.glb",
                f"objects/{model_name}/visual/base{model_id}.glb",
            ]
        )
    missing = [
        name for name in required
        if not (destination_root / name).is_file()
    ]
    if not missing:
        print("RoboTwin L3 object assets already present.", flush=True)
        return

    print(
        "Fetching only the L3 RoboTwin object models needed by these tasks "
        "(range reads; not the multi-gigabyte full archive).",
        flush=True,
    )
    with ZipFile(_HttpRangeFile(ROBOTWIN_OBJECTS_URL)) as archive:
        entries = {info.filename: info for info in archive.infolist()}
        absent = [name for name in missing if name not in entries]
        if absent:
            raise FileNotFoundError("RoboTwin archive is missing: " + ", ".join(absent))
        for index, name in enumerate(missing, start=1):
            target = destination_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entries[name]) as source, target.open("wb") as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            print(f"[{index}/{len(missing)}] {name}", flush=True)


def _download_partnet_model() -> None:
    target = ASSET_ROOT / "partnet_mobility" / "dataset" / "CoffeeMachine" / PARTNET_MODEL
    if (target / "mobility.urdf").is_file():
        print(f"PartNet CoffeeMachine {PARTNET_MODEL} already present.", flush=True)
        return

    archives = [
        item for item in list_files(ASSET_REPO, "dataset")
        if item["path"] == "partnet_mobility.tar.zst"
    ]
    if not archives:
        raise FileNotFoundError("IG-10K-Assets is missing partnet_mobility.tar.zst")
    archive = Path.home() / ".cache/imitator-game-two-task-assets/ig10k/partnet_mobility.tar.zst"
    download_file(ASSET_REPO, "dataset", archives[0], archive)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tar",
            "--use-compress-program=unzstd",
            "--wildcards",
            "--wildcards-match-slash",
            "-xf",
            str(archive),
            "-C",
            str(ASSET_ROOT),
            f"*dataset/CoffeeMachine/{PARTNET_MODEL}/*",
        ],
        check=True,
    )
    if not (target / "mobility.urdf").is_file():
        raise FileNotFoundError(f"PartNet extraction did not produce {target / 'mobility.urdf'}")
    archive.unlink(missing_ok=True)
    print(f"PartNet CoffeeMachine {PARTNET_MODEL} extracted and verified.", flush=True)


def _verify_assets() -> None:
    required = [
        *(ASSET_ROOT / "assets/mani_skill2_ycb/models" / name for name in YCB_MODELS),
        ASSET_ROOT / "partnet_mobility/dataset/CoffeeMachine/103105/mobility.urdf",
        ASSET_ROOT / "partnet_mobility/dataset/CoffeeMachine/103134/mobility.urdf",
    ]
    for model_name, model_id in ROBOTWIN_MODELS.items():
        required.extend(
            [
                ASSET_ROOT / "robotwin/objects" / model_name / f"model_data{model_id}.json",
                ASSET_ROOT / "robotwin/objects" / model_name / "collision" / f"base{model_id}.glb",
                ASSET_ROOT / "robotwin/objects" / model_name / "visual" / f"base{model_id}.glb",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required evaluation assets are missing:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        choices=range(4),
        help="Scene-mismatch levels to download (defaults to all L0-L3).",
    )
    args = parser.parse_args()
    levels = sorted(set(args.levels))

    _write_configs(levels)

    _download_dataset(levels)
    _download_robotwin_models()
    _download_partnet_model()
    _verify_assets()
    print(f"Ready: CleanDesk and PressJuicer, levels {levels}, 50 episodes per branch.")
    print("Data: demos/demo_data (human), demos/robot_data (real robot), demos/imitator_data (simulation)")
    print(f"Configs: {CONFIG_ROOT}")
    print(f"Assets: {ASSET_ROOT}")


if __name__ == "__main__":
    main()
