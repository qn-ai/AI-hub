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
