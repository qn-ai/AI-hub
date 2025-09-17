#!/usr/bin/env python3
"""
Download/copy a Hugging Face model via an Artifactory mirror into local cache,
then copy the snapshot into a project's working directory.

Behavior:
1) If the model snapshot is already present in the local HF cache, skip the network download.
2) Otherwise, download from the HF endpoint (e.g., Artifactory mirror) into local cache.
3) Copy the cached snapshot into the working directory (e.g., ./models/<safe_model_id>).

Environment:
- HF_ENDPOINT  (e.g., https://<your-artifactory>/api/huggingface)
- HF_TOKEN     (required if your mirror requires auth)
- HF_HOME      (optional; to control cache root)
- HUGGINGFACE_HUB_CACHE (optional; another way to control cache)

Usage:
  python scripts/download_model.py --model-id meta-llama/Llama-3-8b --workdir models

Requires:
  pip install huggingface_hub>=0.25
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


def safe_model_dir_name(model_id: str) -> str:
    # Matches HF cache dir naming style; good for project dirs too
    return model_id.replace("/", "--").replace(":", "_")


def copy_snapshot_to_workdir(snapshot_path: Path, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / safe_model_dir_name(snapshot_path.name if snapshot_path.name.startswith("models--") else snapshot_path.name)
    # If snapshot_path is ".../models--<safe>", prefer that <safe> for target name
    if snapshot_path.name.startswith("models--"):
        target = workdir / snapshot_path.name.replace("models--", "", 1)

    # Safer: ensure target is the safe model dir name derived from the model id present in path
    # but keep stable naming:
    target = workdir / safe_model_dir_name(target.name)

    # Copy snapshot contents
    target.mkdir(parents=True, exist_ok=True)
    # Copytree with dirs_exist_ok=True mirrors rsync-like behavior
    for item in snapshot_path.iterdir():
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    return target


def get_cache_dir() -> Path:
    # Respect HF cache env vars if set; otherwise fallback to default
    # huggingface_hub uses HF_HOME or HUGGINGFACE_HUB_CACHE.
    # We don't need to replicate all logic; this is just for informative printing.
    env_cache = os.getenv("HUGGINGFACE_HUB_CACHE") or (
        (Path(os.getenv("HF_HOME")).expanduser() / "hub").as_posix() if os.getenv("HF_HOME") else None
    )
    if env_cache:
        return Path(env_cache).expanduser()
    # Default huggingface_hub cache location
    return Path("~/.cache/huggingface/hub").expanduser()


def resolve_snapshot(
    model_id: str,
    revision: Optional[str],
    cache_dir: Optional[Path],
    allow_network: bool,
    token: Optional[str],
) -> Path:
    """Return the snapshot path in cache. If allow_network=False, raises if not cached."""
    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=not allow_network,
            token=token,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    return snapshot_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="Hugging Face model repo id (e.g., 'bert-base-uncased')")
    parser.add_argument("--revision", default=None, help="Optional revision/commit/tag")
    parser.add_argument("--workdir", default="models", help="Project working directory to copy the model into")
    parser.add_argument("--cache-dir", default=None, help="Optional explicit cache dir (overrides env defaults)")
    parser.add_argument("--force-refresh", action="store_true", help="Force network download even if cached")
    args = parser.parse_args()

    hf_endpoint = os.getenv("HF_ENDPOINT")  # e.g., Artifactory HF proxy URL
    hf_token = os.getenv("HF_TOKEN")
    # huggingface_hub honors HF_ENDPOINT automatically via env var if set.

    if hf_endpoint:
        print(f"[info] Using HF endpoint: {hf_endpoint}")
    if hf_token:
        print("[info] HF token provided via env")

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    if cache_dir:
        print(f"[info] Using explicit cache dir: {cache_dir}")
    else:
        print(f"[info] Using cache dir (effective): {get_cache_dir()}")

    # Step 1: try to resolve from cache only (skip network) unless force-refresh
    snapshot_path = None
    if not args.force_refresh:
        try:
            print("[info] Checking local cache for snapshot...")
            snapshot_path = resolve_snapshot(
                model_id=args.model_id,
                revision=args.revision,
                cache_dir=cache_dir,
                allow_network=False,
                token=hf_token,
            )
            print(f"[ok] Found in cache: {snapshot_path}")
        except LocalEntryNotFoundError:
            print("[info] Not found in cache.")

    # Step 2: if not cached (or force-refresh), download (this will populate cache)
    if snapshot_path is None:
        print("[info] Downloading model into cache...")
        snapshot_path = resolve_snapshot(
            model_id=args.model_id,
            revision=args.revision,
            cache_dir=cache_dir,
            allow_network=True,
            token=hf_token,
        )
        print(f"[ok] Downloaded snapshot: {snapshot_path}")

    # Step 3: copy snapshot contents into working directory
    workdir = Path(args.workdir).expanduser()
    dest = copy_snapshot_to_workdir(Path(snapshot_path), workdir)
    print(f"[done] Copied snapshot to working dir: {dest.resolve()}")


if __name__ == "__main__":
    main()
