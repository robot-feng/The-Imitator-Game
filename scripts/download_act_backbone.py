"""Download the compact pretrained backbone used by the ACT reproduction."""

from pathlib import Path

from hf_download_utils import download_file, list_modelscope_files


MODEL_ID = "facebook/dinov2-base"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
DESTINATION = Path.home() / ".cache" / "imitator-game" / "models" / "facebook" / "dinov2-base"


def main() -> None:
    files = {item["path"]: item for item in list_modelscope_files("model", MODEL_ID)}
    missing = [name for name in REQUIRED_FILES if name not in files]
    if missing:
        raise FileNotFoundError(f"ModelScope {MODEL_ID} is missing: {', '.join(missing)}")

    for name in REQUIRED_FILES:
        download_file(MODEL_ID, "model", files[name], DESTINATION / name)
        print(f"Ready: {DESTINATION / name}", flush=True)

    print(f"Pretrained DINOv2 Base is ready at {DESTINATION}")


if __name__ == "__main__":
    main()
