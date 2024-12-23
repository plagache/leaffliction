import argparse
from PIL import Image
from Augmentor.Operations import Flip, Rotate, Skew, Shear, CropRandom, Distort
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment an image or directories of images")
    parser.add_argument("path", help="the file(s) to augment")
    args = parser.parse_args()
    print(args.path)

    output_directory = "augmented_directory"

    toto = Path(args.path)
    images_paths = []
    input_image = "images/Apple_rust/image (9).JPG"
    images_paths.append(input_image)

    show_image = False
    if toto.is_file():
        show_image = True
        pass
    elif toto.is_dir():
        # Construct set of directories/categories see distribution
        input_image = "images/Apple_rust/image (10).JPG"
        images_paths.append(input_image)
        # extract list of images
        pass
    else:
        pass
        # Exception path error

    #################

    # Create output directories 

    #################
    for image_path in images_paths:
        image = Image.open(input_image)
        # image.show()

        modified_images = {
            "Rotate": Rotate(probability=1, rotation=90).perform_operation([image]),
            "Flip": Flip(probability=1, top_bottom_left_right="RANDOM").perform_operation([image]),
            "Skew": Skew(probability=1, skew_type="TILT", magnitude=1).perform_operation([image]),
            "Shear": Shear(probability=1, max_shear_left=20, max_shear_right=20).perform_operation([image]),
            "Crop": CropRandom(probability=1, percentage_area=0.8).perform_operation([image]),
            "Distortion": Distort(probability=1, grid_width=2, grid_height=2, magnitude=9).perform_operation([image]),
        }

        for modification, images in modified_images.items():
            split_image_name = input_image.split('.')
            split_image_name[0] = f"{split_image_name[0]}_{modification}"
            output_path = '.'.join(split_image_name)
            output_path = f"{output_directory}/{output_path}"
            # output_path.split('/')[:-1]
            print(output_path)
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
