import argparse
from PIL import Image
from Augmentor.Operations import Flip, Rotate, Skew, Shear, CropRandom, Distort
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import DatasetFolder


def get_modified_images(image):
    modified_images = {
        "Rotate": Rotate(probability=1, rotation=90).perform_operation([image]),
        "Flip": Flip(probability=1, top_bottom_left_right="RANDOM").perform_operation([image]),
        "Skew": Skew(probability=1, skew_type="TILT", magnitude=1).perform_operation([image]),
        "Shear": Shear(probability=1, max_shear_left=20, max_shear_right=20).perform_operation([image]),
        "Crop": CropRandom(probability=1, percentage_area=0.8).perform_operation([image]),
        "Distortion": Distort(probability=1, grid_width=2, grid_height=2, magnitude=9).perform_operation([image]),
    }
    return modified_images.items()

def modify_image(image_path, images_to_show):
    image = Image.open(image_path)
    images_to_show.append(("original", image_path))
    for modification, images in get_modified_images(image):
        output_path = get_modified_image_name(modification, image_path)
        images_to_show.append((modification, output_path))
        images[0].save(output_path)


def get_modified_image_name(modification: str, image_path: str) -> str:
    split_image_name = image_path.split(".")
    split_image_name[0] = f"{split_image_name[0]}_{modification}"
    return ".".join(split_image_name)


def display_images(images_with_titles: tuple[str, str]):
    plt.style.use('dark_background')
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
        dataset.modify_images()
        dataset.balance_dataset(output_directory)
        exit(0)
    else:
        pass
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
