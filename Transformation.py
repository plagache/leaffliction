import argparse

import cv2 as cv
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image
# from tinygrad import Tensor, nn
from pathlib import Path

from utils import DatasetFolder, Dataloader
from Augmentation import display_images


def gaussian_blur(numpy_image):
    # opencv_gaussian_blur = cv.GaussianBlur(numpy_image, (5, 5), 0)
    plantcv_gaussian_blur = pcv.gaussian_blur(numpy_image, (5, 5), 0)

    # opencv_image = Image.fromarray(opencv_gaussian_blur)
    # plantcv_image = Image.fromarray(plantcv_gaussian_blur)

    # opencv_image.show()
    # plantcv_image.show()
    return plantcv_gaussian_blur


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
    return pcv_edge


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

def transform_image(image_path, show=False):
    pil_image = Image.open(image_path)
    # pil_image.show()

    numpy_image = np.asarray(pil_image)
    cv_image = cv.imread(image_path)

    # make a tuple of image with their name
    gaussian_image = gaussian_blur(numpy_image)
    canny_image = canny_edge_detection(gaussian_image)
    # canny_image = canny_edge_detection(numpy_image)
    segmentation(numpy_image, cv_image)

    transformed_images = [
        ("gaussian_blur", gaussian_image),
        ("canny_edge_detection", canny_image)
    ]

    if show is True:
        display_images("transformation", transformed_images)
    return transformed_images

def save_transformed(source, image_path, images_with_titles: list[tuple], destination):
    the_end = [part for part in image_path.parts if part not in source.parts]
    new_path = destination / Path(*the_end)

    if new_path.parent.exists() is False:
        new_path.parent.mkdir(parents=True, exist_ok=True)

    for transformation, transformed_image in images_with_titles:
        split_image_name = str(new_path).split(".")
        split_image_name[0] = f"{split_image_name[0]}_{transformation}"
        transformed_filename = ".".join(split_image_name)

        cv.imwrite(transformed_filename, transformed_image)
    return

def transform_dataset(source, destination):
    dataset = DatasetFolder(source)

    # dataloader = Dataloader(dataset, batch_size=2, shuffle=True)
    #
    # for toto in dataloader:
    #     print(toto)
    # exit(0)

    for image_path, class_index in dataset:
        transformed_images = transform_image(image_path)
        save_transformed(source, image_path, transformed_images, destination)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the Transformation of open CV class")
    parser.add_argument("-src", help="the directory to parse from", required=False)
    parser.add_argument("-dst", help="the directory to write to", required=False)
    parser.add_argument("filename", help="a single filename to transform", nargs="?", default=None)
    args = parser.parse_args()


    if args.filename and not (args.src or args.dst):
        print(f"Processing file: {args.filename}")
        # pcv.params.debug = "plot"
        transform_image(args.filename, show=True)
    elif args.src and args.dst and not args.filename:
        print(f"Reading from source: {args.src} and write to destination: {args.dst}")
        # pcv.params.debug = "print"
        # pcv.params.debug_outdir = f"{args.dst}"
        transform_dataset(Path(args.src), Path(args.dst))
    else:
        parser.error("You must provide either a filename or both -src and -dst options.")
        parser.print_help()
