import Augmentor
from pathlib import Path

input_directory = "images"
# input_directory = "images/Apple_Black_rot"
output_directory = "augmented_directory"

cwd = Path.cwd()
output_path = Path(cwd, output_directory, input_directory)
output_directory = output_path.__str__()

Path(output_path).mkdir(parents=True, exist_ok=True)

pipeline = Augmentor.Pipeline(source_directory=input_directory, output_directory=output_directory)

# pipeline.flip_left_right(probability=1)
# pipeline.flip_top_bottom(probability=1)
pipeline.flip_random(probability=1)
pipeline.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
pipeline.skew(probability=1)
pipeline.shear(probability=1, max_shear_left=20, max_shear_right=20)
pipeline.crop_random(probability=1, percentage_area=0.8)
pipeline.random_distortion(probability=1, grid_width=2, grid_height=2, magnitude=9)
# pipeline.zoom_random(probability=1, percentage_area=0.8)
# pipeline.random_contrast( probability=1, min_factor=0.7, max_factor=1.3)
# pipeline.random_color(probability=1, min_factor=0.7, max_factor=1.3)

pipeline.sample(1)

pipeline.process()
