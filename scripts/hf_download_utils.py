"""Small resumable Hugging Face downloader using requests."""

from pathlib import Path
from urllib.parse import quote, urlencode
import hashlib
import time

import requests


HF_ROOT = "https://huggingface.co"
MODELSCOPE_ROOT = "https://modelscope.cn"
MODELSCOPE_MIRRORS = {
    ("dataset", "imitator-game/IG-10K-Dataset"): "Zhouxunzhe/IG-10K-Dataset",
    ("dataset", "imitator-game/IG-10K-Assets"): "Zhouxunzhe/IG-10K-Assets",
}


def _modelscope_session() -> requests.Session:
    # The configured localhost proxy has intermittent disconnects. ModelScope is
    # reachable directly from this host and supports byte-range resume.
    session = requests.Session()
    session.trust_env = False
    return session


def list_modelscope_files(repo_type: str, repo_id: str, subpath: str = "") -> list[dict]:
    session = _modelscope_session()
    kind = "datasets" if repo_type == "dataset" else "models"
    route = "repo/tree" if repo_type == "dataset" else "repo/files"
    url = f"{MODELSCOPE_ROOT}/api/v1/{kind}/{repo_id}/{route}"
    response = session.get(
        url,
        params={
            "Revision": "master",
            "Root": subpath,
            "Recursive": "true",
            "MaxLimit": 10000,
            "FilterDir": "false",
        },
        timeout=(15, 60),
    )
    response.raise_for_status()
    body = response.json()
    if body.get("Code") != 200:
        raise RuntimeError(f"ModelScope listing failed: {body.get('Message')}")
    return [
        {
            "path": item["Path"],
            "size": int(item.get("Size", 0)),
            "sha256": item.get("Sha256"),
            "source": "modelscope",
            "repo_id": repo_id,
            "revision": item.get("Revision") or "master",
        }
        for item in body.get("Data", {}).get("Files", [])
        if item.get("Type") == "blob"
    ]


def _matches_sha256(path: Path, expected: str | None) -> bool:
    if not expected:
        return True
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    return actual.lower() == expected.lower()


def list_files(repo_id: str, repo_type: str, subpath: str = "") -> list[dict]:
    mirror = MODELSCOPE_MIRRORS.get((repo_type, repo_id))
    if mirror:
        try:
            return list_modelscope_files(repo_type, mirror, subpath)
        except Exception as error:
            print(f"ModelScope listing failed for {mirror}; trying Hugging Face: {error}", flush=True)

    kind = "datasets" if repo_type == "dataset" else "models"
    suffix = f"/{quote(subpath, safe='/')}" if subpath else ""
    url = f"{HF_ROOT}/api/{kind}/{repo_id}/tree/main{suffix}"
    last_error = None
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                params={"recursive": "true", "expand": "true"},
                timeout=(15, 45),
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, list):
                raise RuntimeError(f"Unexpected Hugging Face tree response for {repo_id}/{subpath}")
            return [item for item in result if item.get("type") == "file"]
        except requests.RequestException as error:
            last_error = error
            if attempt < 4:
                time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"Could not list {repo_id}/{subpath}") from last_error


def download_file(
    repo_id: str,
    repo_type: str,
    item: dict,
    local_path: Path,
    retries: int = 50,
) -> Path:
    relative = item["path"]
    expected_size = int(item["size"])
    expected_sha256 = item.get("sha256")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.is_file() and local_path.stat().st_size == expected_size:
        if _matches_sha256(local_path, expected_sha256):
            return local_path
        print(f"Checksum mismatch; re-downloading {relative}", flush=True)
        local_path.unlink()

    partial = local_path.with_name(local_path.name + ".part")
    if local_path.exists() and not partial.exists():
        local_path.rename(partial)
    elif local_path.exists():
        local_path.unlink()

    is_modelscope = item.get("source") == "modelscope"
    if is_modelscope:
        actual_repo_id = item["repo_id"]
        kind = "datasets" if repo_type == "dataset" else "models"
        url = f"{MODELSCOPE_ROOT}/api/v1/{kind}/{actual_repo_id}/repo"
        params = {
            "Revision": item.get("revision") or "master",
            "FilePath": relative,
        }
    else:
        kind = "datasets" if repo_type == "dataset" else "models"
        url = f"{HF_ROOT}/{kind}/{repo_id}/resolve/main/{quote(relative, safe='/')}"
        params = None
    session = _modelscope_session() if is_modelscope else requests
    last_error = None
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size or (
            offset == expected_size and not _matches_sha256(partial, expected_sha256)
        ):
            partial.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(
                url,
                params=params,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 180),
            ) as response:
                if response.status_code == 416 and offset == expected_size:
                    partial.replace(local_path)
                    return local_path
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                written = 0
                last_report = 0
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            written += len(chunk)
                            if expected_size >= 256 * 1024**2 and written - last_report >= 256 * 1024**2:
                                print(
                                    f"Downloading {relative}: "
                                    f"{min(offset + written, expected_size) / 1024**3:.2f}/"
                                    f"{expected_size / 1024**3:.2f} GiB",
                                    flush=True,
                                )
                                last_report = written
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise IOError(f"size mismatch: expected {expected_size}, got {actual_size}")
            if not _matches_sha256(partial, expected_sha256):
                partial.unlink(missing_ok=True)
                raise IOError("SHA-256 mismatch")
            partial.replace(local_path)
            return local_path
        except Exception as error:  # keep partial bytes and resume on the next attempt
            last_error = error
            print(f"Retry {attempt + 1}/{retries} {relative}: {type(error).__name__}", flush=True)
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Could not download {relative}") from last_error
