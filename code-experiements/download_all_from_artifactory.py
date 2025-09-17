import os
import requests
from urllib.parse import urljoin

def list_artifactory_files(base_url, repo_path, api_key=None):
    """
    List all files in an Artifactory folder using the REST API.
    Returns a list of file URLs.
    """
    api_url = urljoin(base_url, f"api/storage/{repo_path}")
    headers = {}
    if api_key:
        headers["X-JFrog-Art-Api"] = api_key

    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    data = response.json()
    files = []
    for child in data.get("children", []):
        if not child["folder"]:
            file_url = urljoin(base_url, f"{repo_path}/{child['uri'].lstrip('/')}")
            files.append(file_url)
    return files

def download_file(url, dest_folder, api_key=None):
    """
    Download a single file from Artifactory.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    filename = os.path.basename(url)
    dest_path = os.path.join(dest_folder, filename)
    headers = {}
    if api_key:
        headers["X-JFrog-Art-Api"] = api_key

    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded {filename} to {dest_path}")

def download_all_files_from_artifactory(base_url, repo_path, dest_folder, api_key=None):
    """
    Download all files from an Artifactory folder.
    """
    file_urls = list_artifactory_files(base_url, repo_path, api_key)
    for url in file_urls:
        download_file(url, dest_folder, api_key)

if __name__ == "__main__":
    # Example usage:
    base_url = input("Artifactory base URL (e.g. https://artifactory.mycompany.com/artifactory/): ").strip()
    repo_path = input("Repository path (e.g. my-repo/models/my-model/1.0.0): ").strip()
    dest_folder = input("Local destination folder (default '.'): ").strip() or "."
    api_key = input("API key (leave blank if not needed): ").strip() or None

    download_all_files_from_artifactory(base_url, repo_path, dest_folder, api_key)


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace models with stable directory structure.")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo ID or local path")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory for the model")
    parser.add_argument("--local", action="store_true", help="Use only local cache, no network")
    parser.add_argument("--endpoint", type=str, default=None, help="Custom HuggingFace endpoint")
    parser.add_argument("--token", type=str, default=None, help="Access token for HuggingFace")
    parser.add_argument("--etag_timeout", type=int, default=86400, help="Timeout for etag validation")
    parser.add_argument("--download_timeout", type=int, default=86400, help="Timeout for downloads")

    args = parser.parse_args()

    # Configure environment
    configure_hf_env(
        endpoint=args.endpoint,
        token=args.token,
        etag_timeout=args.etag_timeout,
        download_timeout=args.download_timeout,
    )

    # Download model
    final_dir = download_model_to_dir(
        repo_id_or_path=args.repo,
        dst_dir=Path(args.dst),
        local_files_only=args.local,
    )

    print(f"Model downloaded to: {final_dir}")


if __name__ == "__main__":
    main()
