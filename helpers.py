import urllib.request
from pathlib import Path
from typing import Union
from zipfile import ZipFile


def fetch(url: str, items: Union[Path, str]):
    urllib.request.urlretrieve(url, items)
    return items


def unzip(compressed_dataset, directory):
    directories_name = set()
    zipp = ZipFile(compressed_dataset)
    zipp.extractall(directory)
    print(zipp.filelist)
    for item in zipp.filelist:
        if item.filename.endswith("/"):
            directories_name.add(item.filename)
    return directories_name


if __name__ == "__main__":
    items_url = "https://cdn.intra.42.fr/document/document/17547/leaves.zip"
    compressed_dataset = Path("leaves.zip")
    dataset = Path("images")
    current_directory = Path.cwd()

    if not Path.exists(compressed_dataset):
        fetch(items_url, compressed_dataset)
    if not Path.exists(dataset):
        directories = unzip(compressed_dataset, current_directory)
        print(directories)
        # Path.mkdir(items_directory)

    # os.path.commonprefix()
