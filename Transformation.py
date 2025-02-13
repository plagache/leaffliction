import argparse

import cv2 as cv
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image
# from tinygrad import Tensor, nn

import matplotlib.pyplot as plt
from utils import DatasetFolder, Dataloader


def gaussian_blur(numpy_array):
    # opencv_gaussian_blur = cv.GaussianBlur(numpy_image, (5, 5), 0)
    plantcv_gaussian_blur = pcv.gaussian_blur(numpy_image, (5, 5), 0)

    # opencv_image = Image.fromarray(opencv_gaussian_blur)
    # plantcv_image = Image.fromarray(plantcv_gaussian_blur)

    # opencv_image.show()
    # plantcv_image.show()
    return


def canny_edge_detection(numpy_array):
    pcv_edge = pcv.canny_edge_detect(numpy_array, sigma=2)
    # pcv_image = Image.fromarray(pcv_edge)
    # pcv_image.show()

    # edge = cv.Canny(numpy_array, 200, 300)
    # other_edge = cv.Canny(numpy_array, 150, 250)
    # other_edge_image = Image.fromarray(other_edge)
    # edge_image = Image.fromarray(edge)
    # other_edge_image.show()
    # edge_image.show()
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
    # auto_thresh, _ = pcv.threshold.custom_range(img=img, lower_thresh=[40, 50, 0], upper_thresh=[75, 95, 35], channel="rgb")
    fill_image = pcv.fill_holes(bin_img=auto_thresh)
    roi = pcv.roi.circle(fill_image, x=125, y=125, r=100)
    kept_mask = pcv.roi.filter(mask=fill_image, roi=roi, roi_type="partial")
    analysis_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    # analysis_color = pcv.analyze.color(rgb_img=img, labeled_mask=kept_mask, colorspaces="rgb")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the Transformation of open CV class")
    parser.add_argument("-src", help="the directory to parse from", required=False)
    parser.add_argument("-dst", help="the directory to write to", required=False)
    parser.add_argument("filename", help="a single filename to transform", nargs="?", default=None)
    args = parser.parse_args()

    print(f"found argument {args}")

    if args.filename and not (args.src or args.dst):
        print(f"Processing file: {args.filename}")
    elif args.src and args.dst and not args.filename:
        print(f"Reading from source: {args.src} and write to destination: {args.dst}")
    else:
        parser.error("You must provide either a filename or both -src and -dst options.")
        parser.print_help()

    dataset = DatasetFolder(args.directory)

    dataloader = Dataloader(dataset, batch_size=3, shuffle=True)
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
    one_batch = next(iter(dataloader))
    # print(dataloader.indices)
    # print(one_batch)
    # print(type(one_batch))
    for path, class_index in one_batch:
        # print(path)
        # print(class_index)

        pil_image = Image.open(path)
        # pil_image.show()

        numpy_image = np.asarray(pil_image)
        cv_image = cv.imread(path)

        gaussian_blur(numpy_image)
        canny_edge_detection(numpy_image)
        segmentation(numpy_image, cv_image)
