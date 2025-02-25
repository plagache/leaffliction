import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

from utils import DatasetFolder


def modify_image(image_path, images_to_show):
    image = Image.open(image_path)
    images_to_show.append(("original", image_path))
    for modification, modified_image in DatasetFolder.get_modified_images(image):
        output_path = DatasetFolder.get_modified_image_name(modification, image_path)
        images_to_show.append((modification, output_path))
        modified_image.save(output_path)


def display_images(images_with_titles: tuple[str, str]):
    plt.style.use("dark_background")
    rows = 1
    cols = len(images_with_titles)
    fig, axes = plt.subplots(rows, cols, figsize=(19.2, 10.8))
    fig.suptitle("augmentation", fontsize=16, fontweight="bold")

    for i, (title, path) in enumerate(images_with_titles):
        img = mpimg.imread(path)
        axes[i].imshow(img)
        axes[i].axis("off")

        axes[i].set_title(title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(f"augmented images of {images_with_titles[1][1].split('/')[-1]}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment an image or directories of images")
    parser.add_argument("path", help="the file(s) to augment")
    args = parser.parse_args()

    print("augmenting in", args.path)

    output_directory = "augmented_directory"

    given_path = Path(args.path)

    if given_path.is_file():
        images_to_show: list[tuple[str, str]] = []
        modify_image(args.path, images_to_show)
        display_images(images_to_show)
        exit(0)

    elif given_path.is_dir():
        dataset: DatasetFolder = DatasetFolder(given_path)
        dataset.to_images()
        dataset.augment_images()
        dataset.balance_dataset(output_directory)
        exit(0)
    else:
        raise ValueError("The given path was neither a file nor a directory")
        # Exception path error

    ###################

    # image => apply save and show modified
    # directory => apply and save modified with tree

    # balance dataset
    # 1600
    # 280

    # for range((max - min) % number of transformation):
    #     apply number of transfo ?

    # modified images are always saved next to their original image

    # refacto the modify images loop to work as a function (need to work with only show image)
