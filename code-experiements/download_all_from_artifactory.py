from shutil import copytree, ignore_patterns
from huggingface_hub.utils import model_cache_path
import argparse

def download_and_copy_model(repo_id, hf_token, dst_base_dir=None):
    """
    Downloads a Hugging Face model and copies it to a local directory.

    Args:
        repo_id (str): The Hugging Face repo id (e.g., "username/model-name").
        hf_token (str): Your Hugging Face access token.
        dst_base_dir (str or Path, optional): Destination base directory. Defaults to ~/models/hf/.
    """
    if not hf_token:
        raise ValueError("Please set your Hugging Face token in the hf_token variable.")

    # Determine the expected cache directory for the model
    cache_dir = model_cache_path(repo_id)
    src = Path(cache_dir)

    # If the model is not in cache, download it
    if not src.exists():
        print(f"Model not found in cache. Downloading {repo_id} ...")
        local_dir = snapshot_download(repo_id=repo_id)
        src = Path(local_dir)
    else:
        print(f"Model found in cache at {src}. Skipping download.")

    # Set default destination directory if not provided
    if dst_base_dir is None:
        dst_base_dir = Path.home() / "models" / "hf"
    else:
        dst_base_dir = Path(dst_base_dir)
    dst = dst_base_dir / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Copy the model files to the destination directory, unless it already exists
    if dst.exists():
        print(f"Destination {dst} already exists. Skipping copy.")
    else:
        copytree(src, dst, ignore=ignore_patterns("*.lock"))
        print(f"Copied model files to {dst}")

def main():
    parser = argparse.ArgumentParser(description="Download and copy Hugging Face model to local directory.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id (e.g., username/model-name)")
    parser.add_argument("--hf-token", required=False, help="Hugging Face access token. If not set, uses hf_token variable in script.")
    parser.add_argument("--dst-base-dir", required=False, help="Destination base directory (default: ~/models/hf/)")
    args = parser.parse_args()

    token = args.hf_token if args.hf_token else hf_token
    download_and_copy_model(args.repo_id, token, args.dst_base_dir)

if __name__ == "__main__":
    main()
