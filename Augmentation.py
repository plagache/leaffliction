import argparse
from PIL import Image
from Augmentor.Operations import Flip, Rotate, Skew, Shear, CropRandom, Distort
from pathlib import Path
from Distribution import Category, get_categories_dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment an image or directories of images")
    parser.add_argument("path", help="the file(s) to augment")
    args = parser.parse_args()
    print("working with", args.path)

    output_directory = "augmented_directory"

    given_path = Path(args.path)
    images_paths = []
    output_paths = []

    show_image = False
    if given_path.is_file():
        show_image = False
        images_paths.append(args.path)

        # split by / | remove last one | join back the elements
        path = '/'.join(args.path.split('/')[:-1])

        path = f"{output_directory}/{path}"
        output_paths.append(path)

    elif given_path.is_dir():
        # Construct set of directories/categories see distribution
        categories = get_categories_dic(given_path)
        for category in categories.values():
            path = f"{output_directory}/{category.path}"
            output_paths.append(path)
            for image in category.files:
                # extract list of images
                images_paths.append(f"{category.path}/{image}")

    else:
        pass
        # Exception path error

    # print(output_paths, images_paths[-1])

    # Create output directories
    for path in output_paths:
        Path(path).mkdir(parents=True, exist_ok=True)

    print(f"{len(images_paths)} image to process")
    for count, image_path in enumerate(images_paths, start=1):
        print(f"processing #{count}", image_path)
        image = Image.open(image_path)

        modified_images = {
            "Rotate": Rotate(probability=1, rotation=90).perform_operation([image]),
            "Flip": Flip(probability=1, top_bottom_left_right="RANDOM").perform_operation([image]),
            "Skew": Skew(probability=1, skew_type="TILT", magnitude=1).perform_operation([image]),
            "Shear": Shear(probability=1, max_shear_left=20, max_shear_right=20).perform_operation([image]),
            "Crop": CropRandom(probability=1, percentage_area=0.8).perform_operation([image]),
            "Distortion": Distort(probability=1, grid_width=2, grid_height=2, magnitude=9).perform_operation([image]),
        }

        for modification, images in modified_images.items():
            split_image_name = image_path.split('.')
            split_image_name[0] = f"{split_image_name[0]}_{modification}"
            output_path = '.'.join(split_image_name)
            output_path = f"{output_directory}/{output_path}"
            # print(output_path)
            images[0].save(output_path)
            if show_image is True:
                images[0].show()

    ###################

    # image => apply save and show modified
    # directory => apply and save modified with tree

    # balance dataset
    # 1600
    # 280

    # for range((max - min) % number of transformation):
    #     apply number of transfo ?
