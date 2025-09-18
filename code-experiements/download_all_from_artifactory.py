import os
from huggingface_hub import snapshot_download
from pathlib import Path
from shutil import copytree, ignore_patterns
import argparse
from hf_download import download_and_copy_model

def download_and_copy_model(

        repo_id (str): The Hugging Face repository ID (e.g., "username/model-name").
        dst_base_dir (str or Path, optional): Base directory to copy the downloaded model to.
            Defaults to '~/models/hf/' if not specified.
        endpoint (str, optional): The Hugging Face endpoint URL. Defaults to "https://huggingface.co".
        etag_timeout (str, optional): Timeout (in seconds) for ETag caching. Defaults to "86400".
        download_timeout (str, optional): Timeout (in seconds) for model download. Defaults to "86400".

    Raises:
        ValueError: If the Hugging Face token is not provided.

    Notes:
        - The function sets environment variables for the Hugging Face Hub to control timeouts and endpoint.
        - Uses `snapshot_download` from `huggingface_hub` to download the model.
        - Copies the downloaded model files to the specified destination directory.
        - If the destination already exists, the copy is skipped.

    Example:
        ```python

        download_and_copy_model(
            repo_id="bert-base-uncased",
            hf_token="your_hf_token_here",
            dst_base_dir="/path/to/local/models",
            etag_timeout="3600",
            download_timeout="3600"
        )
        ```
    repo_id,
    hf_token,
    dst_base_dir=None,
    endpoint="https://huggingface.co",
    etag_timeout="86400",
    download_timeout="86400"
):
    """
    Downloads a Hugging Face model and copies it to a local directory.

    Args:
        repo_id (str): The Hugging Face repo id (e.g., "username/model-name").
        hf_token (str): Your Hugging Face access token.
        dst_base_dir (str or Path, optional): Destination base directory. Defaults to ~/models/hf/.
        endpoint (str): Hugging Face endpoint URL.
        etag_timeout (str): ETag timeout in seconds.
        download_timeout (str): Download timeout in seconds.
    """
    if not hf_token:
        raise ValueError("Please set your Hugging Face token in the hf_token variable or pass it as an argument.")

    # Set environment variables
    os.environ["HF_HUB_ETAG_TIMEOUT"] = etag_timeout
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = download_timeout
    os.environ["HF_HUB_TOKEN"] = hf_token
    os.environ["HF_ENDPOINT"] = endpoint

    # Download the model using snapshot_download
    print(f"Downloading {repo_id} ...")
    local_dir = snapshot_download(repo_id=repo_id, token=hf_token)
    src = Path(local_dir)

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
    parser.add_argument("--hf-token", required=True, help="Hugging Face access token.")
    parser.add_argument("--dst-base-dir", required=False, help="Destination base directory (default: ~/models/hf/)")
    parser.add_argument("--endpoint", required=False, default="https://huggingface.co", help="Hugging Face endpoint URL.")
    parser.add_argument("--etag-timeout", required=False, default="86400", help="ETag timeout in seconds.")
    parser.add_argument("--download-timeout", required=False, default="86400", help="Download timeout in seconds.")
    args = parser.parse_args()

    download_and_copy_model(
        args.repo_id,
        args.hf_token,
        args.dst_base_dir,
        args.endpoint,
        args.etag_timeout,
        args.download_timeout
    )

if __name__ == "__main__":
    main()

    
