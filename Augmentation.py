import argparse
from  dataclasses import dataclass
from PIL import Image
from Augmentor.Operations import Flip, Rotate, Skew, Shear, CropRandom, Distort
from pathlib import Path
from shutil import copy
from Distribution import Category, get_categories
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

def modify_images(images_paths, images_to_show):
    modified_images = []
    for count, image_path in enumerate(images_paths, start=1):
        print(f"processing #{count}", image_path)
        # todo add tqdm progress bar
        image = Image.open(image_path)

        if show_image is True:
            images_to_show.append(("original", image_path))

        for modification, images in get_modified_images(image):
            output_path = get_modified_image_name(modification, image_path)
            if show_image is True:
                images_to_show.append((modification, output_path))
            modified_images.append(output_path)
            images[0].save(output_path)
    return modified_images


def get_modified_image_name(modification, image_path):
    split_image_name = image_path.split(".")
    split_image_name[0] = f"{split_image_name[0]}_{modification}"
    return ".".join(split_image_name)


def display_images(images_with_titles):
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
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment an image or directories of images")
    parser.add_argument("path", help="the file(s) to augment")
    args = parser.parse_args()
    print("working with", args.path)

    output_directory = "augmented_directory"

    given_path = Path(args.path)

    max_count = -1
    show_image = False
    images_to_show = []
    categories: list[Category] = None

    if given_path.is_file():
        show_image = True
        modify_images([args.path], images_to_show)
        display_images(images_to_show)
        exit(0)

    elif given_path.is_dir():
        # Construct set of directories/categories see distribution
        categories = get_categories(given_path)
        for category in categories:
            max_count = category.count if category.count > max_count else max_count

            category.files = [ f"{category.path}/{image}" for image in category.files ]
            category.new_path = f"{output_directory}/{category.path}"

    else:
        pass
        # Exception path error

    # Modify images
    for category in categories:
        category.modified_images = modify_images(category.files, [])

    print("Images have been modified\nNow balancing the dataset")
    # Create output directories (only relevant for balancing dataset)
    for category in categories:
        Path(category.new_path).mkdir(parents=True, exist_ok=True)
        # when dir is created we can then balance this category around the biggest known
        # 1 - copy all original images in new_path
        for file in category.files:
            copy(file, category.new_path)

        # 2 - copy max_count - category.count images in new_path
        if max_count > category.count:
            # todo add a shuffle of modified_images to select them at random
            for file in category.modified_images[:max_count - category.count]:
                copy(file, category.new_path)

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
