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
image.show()

rotated_image = Rotate(probability=1, rotation=90).perform_operation([image])
fliped_image = Flip(probability=1, top_bottom_left_right="RANDOM").perform_operation([image])
skewed_image = Skew(probability=1, skew_type="TILT", magnitude=1).perform_operation([image])
sheared_image = Shear(probability=1, max_shear_left=20, max_shear_right=20).perform_operation([image])
croped_image = CropRandom(probability=1, percentage_area=0.8).perform_operation([image])
distorted_image = Distort(probability=1, grid_width=2, grid_height=2, magnitude=9).perform_operation([image])

distorted_image[0].show()
croped_image[0].show()
sheared_image[0].show()
fliped_image[0].show()
rotated_image[0].show()
skewed_image[0].show()

# balance dataset
# 1600
# 280

# for range((max - min) % number of transformation):
#     apply number of transfo ?
