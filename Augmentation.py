import argparse
from PIL import Image
from Augmentor.Operations import Flip, Rotate, Skew, Shear, CropRandom, Distort
from pathlib import Path
from Distribution import Category, get_categories_dic
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


def get_modified_image_name(modification, image_path):
    split_image_name = image_path.split(".")
    split_image_name[0] = f"{split_image_name[0]}_{modification}"
    return ".".join(split_image_name)


def display_images(images_with_titles):
    plt.style.use('dark_background')
    rows = 1
    cols = len(images_with_titles)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    fig.suptitle("augmentation", fontsize=16, fontweight="bold")

    for i, (title, path) in enumerate(images_with_titles):
        print(image)
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
    images_paths = []
    output_paths = []

    max_count = -1
    show_image = False
    show_images = []

    if given_path.is_file():
        show_image = True
        images_paths.append(args.path)

        # split by / | remove last one | join back the elements
        path = "/".join(args.path.split("/")[:-1])

        path = f"{output_directory}/{path}"
        output_paths.append(path)

    elif given_path.is_dir():
        # Construct set of directories/categories see distribution
        categories: dict[str, Category] = get_categories_dic(given_path)
        for category in categories.values():
            max_count = category.count if category.count > max_count else max_count

            path = f"{output_directory}/{category.path}"
            output_paths.append(path)
            for image in category.files:
                # extract list of images
                images_paths.append(f"{category.path}/{image}")

    else:
        pass
        # Exception path error

    # Create output directories
    for path in output_paths:
        Path(path).mkdir(parents=True, exist_ok=True)

    # Modify images
    print(f"{len(images_paths)} image to process")
    for count, image_path in enumerate(images_paths, start=1):
        print(f"processing #{count}", image_path)
        image = Image.open(image_path)

        original_output_path = f"{output_directory}/{image_path}"
        show_images.append(("original", original_output_path))
        image.save(original_output_path)

        for modification, images in get_modified_images(image):
            output_path = f"{output_directory}/{get_modified_image_name(modification, image_path)}"
            show_images.append((modification, output_path))
            images[0].save(output_path)

    if show_image is True:
        display_images(show_images)

    ###################

    # image => apply save and show modified
    # directory => apply and save modified with tree

    # balance dataset
    # 1600
    # 280

    # for range((max - min) % number of transformation):
    #     apply number of transfo ?
