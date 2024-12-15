from PIL import Image
from Augmentor.Operations import Flip, Rotate, Skew, Shear, CropRandom, Distort
from pathlib import Path

# input_directory = "images"
input_image = "images/Apple_rust/image (9).JPG"
output_directory = "augmented_directory"

cwd = Path.cwd()
# output_path = Path(cwd, output_directory, input_directory)
# output_directory = output_path.__str__()

# Path(output_path).mkdir(parents=True, exist_ok=True)

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
    images[0].save('.'.join(split_image_name))
    images[0].show()

# balance dataset
# 1600
# 280

# for range((max - min) % number of transformation):
#     apply number of transfo ?
