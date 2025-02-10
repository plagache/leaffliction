import argparse

import cv2 as cv
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image
# from tinygrad import Tensor, nn

from utils import DatasetFolder


def gaussian_blur(numpy_array):
    # Gaussian Blur
    opencv_gaussian_blur = cv.GaussianBlur(numpy_image, (5, 5), 0)
    plantcv_gaussian_blur = pcv.gaussian_blur(numpy_image, (5, 5), 0)

    opencv_image = Image.fromarray(opencv_gaussian_blur)
    plantcv_image = Image.fromarray(plantcv_gaussian_blur)

    opencv_image.show()
    plantcv_image.show()
    return


def canny_edge_detection(numpy_array):
    edge = cv.Canny(numpy_array, 150, 250)
    edge_image = Image.fromarray(edge)
    edge_image.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the Transformation of open CV class")
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    # dataset = DatasetFolder(Path(args.directory))
    dataset = DatasetFolder(args.directory)

    # i do not want to compute all images at each run
    path, class_index = dataset[2]
    pil_image = Image.open(path)
    numpy_image = np.asarray(pil_image)

    gaussian_blur(numpy_image)
    canny_edge_detection(numpy_image)

    pil_image.show()
