"""Download the compact CleanDesk + PressJuicer L0 LeRobot subset."""

from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from hf_download_utils import download_file, list_files


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "demos"

HUMAN = ["human_H19", "human_H50"]
ROBOT = ["robot_H19_L0", "robot_H50_L0"]
SIM = ["L0_TwoRobotCleanDesk-v1", "L0_TwoRobotPressJuicer-v1"]
DATASET_LOCAL_ROOTS = {
    "imitator_human_v1/": "demo_data/",
    "imitator_robot_v1/": "robot_data/",
    "imitator_sim_v1_zed2i/": "imitator_data/",
}


def local_dataset_path(remote_path: str) -> Path:
    for remote_root, local_root in DATASET_LOCAL_ROOTS.items():
        if remote_path.startswith(remote_root):
            return DATA_ROOT / local_root / remote_path[len(remote_root) :]
    raise ValueError(f"Unrecognized IG-10K dataset path: {remote_path}")


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
    print("Data roots: demo_data (human), robot_data (real robot), imitator_data (simulation)")
    print("Selected branches: human H19/H50, robot H19/H50 L0, sim CleanDesk/PressJuicer L0")
    print("Human RGB: ZED2i only; MANO and actions are in the selected parquet files.")


if __name__ == "__main__":
    main()
