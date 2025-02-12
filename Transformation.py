import argparse

import cv2 as cv
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image
# from tinygrad import Tensor, nn

import matplotlib.pyplot as plt
from utils import DatasetFolder, Dataloader


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
    binary_mask = pcv.threshold.binary(gray_img=hsv_img, threshold=110, object_type="dark")
    # binary_mask = pcv.threshold.binary(gray_img=hsv_img, threshold=100, object_type="light")

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
    return


def segmentation(numpy_array, img):
    # analyse the colorspaces
    colorspace_img = pcv.visualize.colorspaces(rgb_img=numpy_array, original_img=False)
    # choose a channel that correctly separate background and the leaf
    grayscale_img = pcv.rgb2gray_hsv(rgb_img=numpy_array, channel="s")
    # grayscale_img = pcv.rgb2gray_lab(rgb_img=numpy_array, channel="a")
    # plot the histogram of the grayscale values
    # hist = pcv.visualize.histogram(img=grayscale_img, bins=30)
    # plt.figure()
    # plt.hist(grayscale_img.ravel(), bins=30, range=[0, 255])
    # plt.title('grayscale image histogram')
    # plt.xlabel('pixel intensity')
    # plt.ylabel('Frequency')
    # plt.show()
    # set_thresh = pcv.threshold.binary(gray_img=grayscale_img, threshold=80, object_type="light")
    auto_thresh = pcv.threshold.otsu(gray_img=grayscale_img, object_type="light")
    fill_image = pcv.fill_holes(bin_img=auto_thresh)
    roi = pcv.roi.circle(fill_image, x=125, y=125, r=100)
    kept_mask  = pcv.roi.filter(mask=fill_image, roi=roi, roi_type='partial')
    analysis_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the Transformation of open CV class")
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    # dataset = DatasetFolder(Path(args.directory))
    dataset = DatasetFolder(args.directory)

    dataloader = Dataloader(dataset, batch_size=5, shuffle=True)
    # print(dataloader.indices)
    # Add the plotting of all transformation with plantcv
    pcv.params.debug = "plot"

    # THIS WILL BE A TEST CASE IN TEST_UTILS.PY
    # one_batch = next(iter(dataloader))
    # # print(dataloader.indices)
    # print(one_batch)
    #
    # dataloader.shuffle = False
    #
    # one_batch = next(iter(dataloader))
    # # print(dataloader.indices)
    # print(one_batch)

    # i do not want to compute all images at each run
    path, class_index = dataset[2]
    print(path)
    pil_image = Image.open(path)
    numpy_image = np.asarray(pil_image)
    cv_image = cv.imread(path)

    # gaussian_blur(numpy_image)
    # canny_edge_detection(numpy_image)
    # roi_detection(numpy_image, cv_image)
    # testing_mask(numpy_image, cv_image)
    # testing_binary_mask(numpy_image, cv_image)
    segmentation(numpy_image, cv_image)

    # pil_image.show()
