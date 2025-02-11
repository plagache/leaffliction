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
    pcv_edge = pcv.canny_edge_detect(numpy_array, sigma=2)
    pcv_image = Image.fromarray(pcv_edge)
    edge = cv.Canny(numpy_array, 200, 300)
    other_edge = cv.Canny(numpy_array, 150, 250)
    other_edge_image = Image.fromarray(other_edge)
    edge_image = Image.fromarray(edge)
    other_edge_image.show()
    edge_image.show()
    pcv_image.show()
    return


def roi_detection(numpy_array, cv_image):
    colorspace_img = pcv.visualize.colorspaces(rgb_img=numpy_array)
    l_gray = pcv.rgb2gray_lab(rgb_img=numpy_array, channel="l")
    bin_mask = pcv.threshold.otsu(gray_img=l_gray, object_type="light")
    cleaner_mask = pcv.fill(bin_img=bin_mask, size=50)
    clean_mask = pcv.fill_holes(cleaner_mask)
    roi1 = pcv.roi.circle(numpy_array, x=125, y=125, r=100)
    # kept_mask = pcv.roi.filter(mask=a_fill_image, roi=roi1, roi_type="partial")
    # roi = pcv.roi.filter(mask=bin_mask, roi=roi, roi_type="partial")
    return


def testing_mask(numpy_array, img):
    # Convert the image to HSV color space
    hsv_img = pcv.rgb2gray_hsv(rgb_img=img, channel="h")

    # Apply a threshold to isolate the leaf
    binary_mask = pcv.threshold.binary(gray_img=hsv_img, threshold=100, object_type="light")

    # Apply the mask
    masked_image = pcv.apply_mask(img=img, mask=binary_mask, mask_color="white")

    # Apply morphological operations
    cleaned_mask = pcv.morphology.erode(mask=binary_mask, ksize=3)
    cleaned_mask = pcv.morphology.dilate(mask=cleaned_mask, ksize=3)
    return


def testing_binary_mask(numpy_array, img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

    mask = binary

    masked_img = cv.bitwise_and(img, img, mask=mask)

    cv.imshow("Original Image", img)
    cv.imshow("Grayscale Image", gray)
    cv.imshow("Binary Image", binary)
    cv.imshow("Masked Image", masked_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the Transformation of open CV class")
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    # dataset = DatasetFolder(Path(args.directory))
    dataset = DatasetFolder(args.directory)

    # Add the plotting of all transformation with plantcv
    pcv.params.debug = "plot"

    # i do not want to compute all images at each run
    path, class_index = dataset[2]
    print(path)
    pil_image = Image.open(path)
    numpy_image = np.asarray(pil_image)
    cv_image = cv.imread(path)

    # gaussian_blur(numpy_image)
    # canny_edge_detection(numpy_image)
    # roi_detection(numpy_image, cv_image)
    testing_binary_mask(numpy_image, cv_image)
    testing_mask(numpy_image, cv_image)

    # pil_image.show()
