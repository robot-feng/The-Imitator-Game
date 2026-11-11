"""Download the compact CleanDesk + PressJuicer L0 LeRobot subset."""

from pathlib import Path
from pathlib import PurePosixPath
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from hf_download_utils import download_file, list_files


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset"

HUMAN = ["human_H19", "human_H50"]
ROBOT = ["robot_H19_L0", "robot_H50_L0"]
SIM = ["L0_TwoRobotCleanDesk-v1", "L0_TwoRobotPressJuicer-v1"]
TASKS = {
    "CleanDesk": {"human": "human_H19", "robot": "robot_H19"},
    "PressJuicer": {"human": "human_H50", "robot": "robot_H50"},
}


def local_dataset_path(remote_path: str) -> Path:
    parts = PurePosixPath(remote_path).parts
    if len(parts) < 2:
        raise ValueError(f"Unrecognized IG-10K dataset path: {remote_path}")
    namespace, repo_id, *relative_path = parts

    if namespace == "imitator_human_v1":
        task = next((name for name, ids in TASKS.items() if ids["human"] == repo_id), None)
        if task is None:
            raise ValueError(f"Unrecognized human dataset: {repo_id}")
        local_root = Path("human_data") / task
    elif namespace == "imitator_robot_v1":
        try:
            repo_prefix, level = repo_id.rsplit("_L", 1)
            level = int(level)
        except (ValueError, TypeError):
            raise ValueError(f"Unrecognized robot dataset: {repo_id}") from None
        task = next((name for name, ids in TASKS.items() if ids["robot"] == repo_prefix), None)
        if task is None or level != 0:
            raise ValueError(f"Unrecognized L0 robot dataset: {repo_id}")
        local_root = Path("robot_data") / task / f"L{level}"
    elif namespace == "imitator_sim_v1_zed2i":
        if not repo_id.startswith("L0_TwoRobot") or not repo_id.endswith("-v1"):
            raise ValueError(f"Unrecognized L0 simulation dataset: {repo_id}")
        task = repo_id[len("L0_TwoRobot") : -len("-v1")]
        if task not in TASKS:
            raise ValueError(f"Unrecognized L0 simulation task: {task}")
        local_root = Path("sim_data") / task / "L0"
    else:
        raise ValueError(f"Unrecognized IG-10K dataset namespace: {namespace}")

    return DATA_ROOT / local_root / Path(*relative_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-only", action="store_true", help="Download human videos and metadata only")
    args = parser.parse_args()

    branches = [f"imitator_human_v1/{repo}" for repo in HUMAN]
    if not args.human_only:
        branches.extend(f"imitator_robot_v1/{repo}" for repo in ROBOT)
        branches.extend(f"imitator_sim_v1_zed2i/{repo}" for repo in SIM)

    folders = [f"{branch}/{subpath}" for branch in branches
               for subpath in ("data", "meta", "videos/observation.images.zed2i")]
    sim_roots = [branch for branch in branches if branch.startswith("imitator_sim_v1_zed2i/")]
    selected = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(list_files, "imitator-game/IG-10K-Dataset", "dataset", folder): folder
            for folder in folders + sim_roots
        }
        for future in as_completed(futures):
            folder = futures[future]
            for item in future.result():
                if folder in sim_roots and item["path"] != f"{folder}/.complete":
                    continue
                selected[item["path"]] = item

    total_bytes = sum(int(item["size"]) for item in selected.values())
    print(f"Downloading {len(selected)} files ({total_bytes / 1024**3:.2f} GiB) with two workers.")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(
                download_file,
                "imitator-game/IG-10K-Dataset",
                "dataset",
                item,
                local_dataset_path(item["path"]),
            ): item
            for item in selected.values()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            future.result()
            print(f"[{index}/{len(selected)}] {item['path']} ({int(item['size']) / 1024**2:.1f} MiB)")

    print(f"Subset downloaded under {DATA_ROOT}")
    print("Data roots: human_data, robot_data, sim_data; each is grouped by task")
    print("Selected branches: human H19/H50, robot H19/H50 L0, sim CleanDesk/PressJuicer L0")
    print("Human RGB: ZED2i only; MANO and actions are in the selected parquet files.")


if __name__ == "__main__":
    main()
