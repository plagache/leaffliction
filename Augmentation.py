import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from utils import DatasetFolder, get_modified_image_name, get_modified_images


def modify_image(image_path: Path, images_to_show: list) -> None:
    image = Image.open(image_path)
    images_to_show.append(("original", image))
    for modification, modified_image in get_modified_images(image):
        output_path: str = get_modified_image_name(modification, image_path)
        images_to_show.append((modification, modified_image))
        modified_image.save(output_path)


def display_images(title: str, images_with_titles: list[tuple], show=True) \
        -> None:
    plt.style.use("dark_background")
    count = len(images_with_titles)
    max_cols = 5
    cols = count if count < max_cols else max_cols
    rows = count // max_cols
    if count % max_cols != 0:
        rows += 1
    fig, axes = plt.subplots(rows, cols, figsize=(19.2, 10.8))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    # axes = axes.flat
    flatten_axes = axes.flatten()

    for axe in flatten_axes:
        axe.axis("off")

    for (title, img), axe in zip(images_with_titles, flatten_axes):
        axe.imshow(img)
        axe.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if show is True:
        plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Augment an image or directories of images")
    parser.add_argument("path", help="the file(s) to augment")
    args = parser.parse_args()

    output_directory = "augmented_directory"

    given_path = Path(args.path)
    print(f"augmenting {given_path}")

    if given_path.is_file():
        images_to_show: list[tuple[str, str]] = []
        modify_image(given_path, images_to_show)
        display_images("augmentation", images_to_show)
        exit(0)

    elif given_path.is_dir():
        dataset: DatasetFolder = DatasetFolder(given_path)
        dataset.to_images()
        dataset.augment_images()
        dataset.balance_dataset(output_directory)
        exit(0)
    else:
        raise ValueError("The given path was neither a file nor a directory")
