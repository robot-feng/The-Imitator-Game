"""Fetch only the simulation assets required by CleanDesk and PressJuicer L0."""

from pathlib import Path, PurePosixPath
import subprocess
import zipfile

from hf_download_utils import download_file, list_files


ASSET_ROOT = Path.home() / ".maniskill" / "data"
DOWNLOAD_ROOT = Path.home() / ".cache" / "imitator-game-two-task-assets"
YCB_IDS = ("026_sponge", "061_foam_brick")
PARTNET_ID = "103105"


def extract_zstd(archive: Path, members: list[str] | None = None) -> None:
    command = [
        "tar", "--use-compress-program=unzstd", "--wildcards",
        "--wildcards-match-slash", "-xf", str(archive), "-C", str(ASSET_ROOT),
    ]
    if members:
        command.extend(members)
    subprocess.run(command, check=True)


def safe_zip_target(name: str, root: Path) -> Path | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = list(path.parts)
    try:
        base_index = parts.index("mani_skill2_ycb")
    except ValueError:
        return None
    relative = Path(*parts[base_index + 1 :])
    return root / relative


def extract_ycb_subset(archive: Path) -> None:
    target_root = ASSET_ROOT / "assets" / "mani_skill2_ycb"
    wanted = set(YCB_IDS)
    with zipfile.ZipFile(archive) as zf:
        selected = []
        for info in zf.infolist():
            path = PurePosixPath(info.filename)
            is_metadata = path.name == "info_pick_v0.json"
            is_selected_model = any(
                "models" in path.parts and model_id in path.parts for model_id in wanted
            )
            if is_metadata or is_selected_model:
                selected.append((info, safe_zip_target(info.filename, target_root)))
        if not any(PurePosixPath(info.filename).name == "info_pick_v0.json" for info, _ in selected):
            raise RuntimeError("YCB archive did not contain info_pick_v0.json")
        for info, target in selected:
            if target is None or info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    asset_files = {
        item["path"]: item
        for item in list_files("imitator-game/IG-10K-Assets", "dataset")
        if item["path"] in {"assets.tar.zst", "partnet_mobility.tar.zst"}
    }
    if set(asset_files) != {"assets.tar.zst", "partnet_mobility.tar.zst"}:
        raise RuntimeError("IG-10K-Assets repository is missing a required archive")
    assets_archive = DOWNLOAD_ROOT / "ig10k" / "assets.tar.zst"
    partnet_archive = DOWNLOAD_ROOT / "ig10k" / "partnet_mobility.tar.zst"
    download_file("imitator-game/IG-10K-Assets", "dataset", asset_files["assets.tar.zst"], assets_archive)
    download_file("imitator-game/IG-10K-Assets", "dataset", asset_files["partnet_mobility.tar.zst"], partnet_archive)
    extract_zstd(assets_archive)
    extract_zstd(
        partnet_archive,
        [f"*dataset/CoffeeMachine/{PARTNET_ID}/*"],
    )

    ycb_files = list_files("haosulab/ManiSkill2", "dataset", "data")
    ycb_item = next((item for item in ycb_files if item["path"] == "data/mani_skill2_ycb.zip"), None)
    if ycb_item is None:
        raise RuntimeError("ManiSkill2 repository is missing data/mani_skill2_ycb.zip")
    ycb_archive = DOWNLOAD_ROOT / "ycb" / "data" / "mani_skill2_ycb.zip"
    download_file("haosulab/ManiSkill2", "dataset", ycb_item, ycb_archive)
    extract_ycb_subset(ycb_archive)

    expected = [
        ASSET_ROOT / "partnet_mobility" / "dataset" / "CoffeeMachine" / PARTNET_ID / "mobility.urdf",
        ASSET_ROOT / "assets" / "mani_skill2_ycb" / "info_pick_v0.json",
        *(ASSET_ROOT / "assets" / "mani_skill2_ycb" / "models" / model_id for model_id in YCB_IDS),
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Asset extraction incomplete:\n" + "\n".join(missing))

    # The source archives are reproducible and remain in the Hugging Face cache.
    # Remove only the local download copies after the selected assets are verified.
    for archive in (assets_archive, partnet_archive, ycb_archive):
        archive.unlink(missing_ok=True)
    print("CleanDesk + PressJuicer L0 simulation assets are ready under", ASSET_ROOT)
    print("YCB:", ", ".join(YCB_IDS), "| PartNet CoffeeMachine:", PARTNET_ID)


if __name__ == "__main__":
    main()
