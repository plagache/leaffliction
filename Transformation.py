import argparse

import cv2 as cv
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image
# from tinygrad import Tensor, nn
from pathlib import Path
from tqdm import tqdm

from utils import DatasetFolder, Dataloader
from Augmentation import display_images

def apply_gaussian(image):
    gaussian_3_3 = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    output = cv.filter2D(image, -1, gaussian_3_3)
    # gaussian_5_5 = (1 / 159) * np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])
    # output = cv.filter2D(image, -1, gaussian_5_5)
    return output


def apply_vertical(image):
    sobel_y_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = cv.filter2D(image, cv.CV_32F, sobel_y_kernel)
    return output


def apply_horizontal(image):
    sobel_x_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = cv.filter2D(image, cv.CV_32F, sobel_x_kernel)
    return output


def combine_edge_detection(vertical_image, horizontal_image):
    threshold_value = 75
    # output = np.sqrt(np.square(vertical_image) + np.square(horizontal_image))
    combine_gradients = np.abs(vertical_image) + np.abs(horizontal_image)

    normalized_magnitude_image = cv.normalize(combine_gradients, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    binary_image = np.zeros_like(combine_gradients, dtype=np.uint8)

    binary_image[normalized_magnitude_image > threshold_value] = 1
    return combine_gradients, binary_image


def segmenting_red_green(numpy_array):
    # lab color space: luminosity, Red/Green, Yellow/Blue
    a_channel = pcv.rgb2gray_lab(rgb_img=numpy_array, channel='a')
    # median blur use the median: remove salt and paper
    median_img = pcv.median_blur(gray_img=a_channel, ksize=11)
    otsu = pcv.threshold.otsu(gray_img=median_img, object_type="dark")
    # gaussian blur calculate the mean of neighbor: produce smooth and soft edges
    gaussian_img = pcv.gaussian_blur(img=otsu, ksize=(5, 5), sigma_x=0, sigma_y=None)
    leaf_mask = pcv.fill_holes(bin_img=otsu)
    masked = pcv.apply_mask(img=numpy_array, mask=leaf_mask, mask_color='white')
    return a_channel, median_img, otsu, gaussian_img, leaf_mask, masked


def segmenting_blue_yellow(numpy_array):
    # lab color space: luminosity, Red/Green, Yellow/Blue
    b_channel = pcv.rgb2gray_lab(rgb_img=numpy_array, channel='b')
    # median blur use the median: remove salt and paper
    median_img = pcv.median_blur(gray_img=b_channel, ksize=11)
    otsu = pcv.threshold.otsu(gray_img=median_img, object_type="light")
    # gaussian blur calculate the mean of neighbor: produce smooth and soft edges
    # cannot use gaussian as a binary since it create gray point
    gaussian_img = pcv.gaussian_blur(img=otsu, ksize=(5, 5), sigma_x=0, sigma_y=None)
    leaf_mask = pcv.fill_holes(bin_img=otsu)
    masked = pcv.apply_mask(img=numpy_array, mask=leaf_mask, mask_color='white')
    return b_channel, median_img, otsu, gaussian_img, leaf_mask, masked


def segmenting_saturation(numpy_array):
    s_channel = pcv.rgb2gray_hsv(rgb_img=numpy_array, channel="s")
    # median blur use the median: remove salt and paper
    median_img = pcv.median_blur(gray_img=s_channel, ksize=11)
    otsu = pcv.threshold.otsu(gray_img=median_img, object_type="light")
    # gaussian blur calculate the mean of neighbor: produce smooth and soft edges
    gaussian_img = pcv.gaussian_blur(img=otsu, ksize=(5, 5), sigma_x=0, sigma_y=None)
    leaf_mask = pcv.fill_holes(bin_img=otsu)
    masked = pcv.apply_mask(img=numpy_array, mask=leaf_mask, mask_color='white')
    return s_channel, median_img, otsu, gaussian_img, leaf_mask, masked


def transform_image(image_path):
    pil_image = Image.open(image_path)
    # pil_image.show()

    numpy_image = np.asarray(pil_image)
    gray_image = cv.cvtColor(numpy_image, cv.COLOR_BGR2GRAY)

    gaussian_image = apply_gaussian(gray_image)

    vertical_edges = apply_vertical(gaussian_image)
    horizontal_edges = apply_horizontal(gaussian_image)

    combine_gradients, combine_edges = combine_edge_detection(vertical_edges, horizontal_edges)
    combine_edges = apply_gaussian(combine_edges)
    combine_gradients = apply_gaussian(combine_gradients)

    a_channel, median_img, otsu, gaussian_img, leaf_mask_a, masked = segmenting_red_green(numpy_image)
    b_channel, median_img, otsu, gaussian_img, leaf_mask_b, masked = segmenting_blue_yellow(numpy_image)
    s_channel, median_img, otsu, gaussian_img, leaf_mask_s, masked = segmenting_saturation(numpy_image)

    leaf_mask_c = pcv.logical_or(bin_img1=leaf_mask_a, bin_img2=leaf_mask_b)
    leaf_mask_d = pcv.logical_or(bin_img1=leaf_mask_s, bin_img2=leaf_mask_b)
    leaf_mask_z = pcv.logical_or(bin_img1=leaf_mask_c, bin_img2=leaf_mask_d)

    masked = pcv.apply_mask(img=numpy_image, mask=leaf_mask_c, mask_color='white')
    # masked = pcv.apply_mask(img=numpy_image, mask=leaf_mask_z, mask_color='white')

    # roi = pcv.roi.circle(img=masked, x=125, y=125, r=100)
    # filtered_mask = pcv.roi.filter(mask=leaf_mask_z, roi=roi, roi_type='partial')
    pcv.params.line_thickness = 2
    size_analysis_image = pcv.analyze.size(img=numpy_image, labeled_mask=leaf_mask_z)

    transformed_images = [
        ("Original", numpy_image),
        ("Gaussian Blur", gaussian_image),
        ("Vertical Edges", vertical_edges),
        ("Horizontal Edges", horizontal_edges),
        ("Combine Gradient", combine_gradients),
        ("Combine Edges", combine_edges),
        # ("S channel", s_channel),
        ("B channel", b_channel),
        ("A channel", a_channel),
        # ("median blur", median_img),
        # ("otsu", otsu),
        # ("gaussian_img", gaussian_img),
        # ("leaf mask b", leaf_mask_b),
        # ("leaf mask a", leaf_mask_a),
        # ("leaf mask c", leaf_mask_c),
        # ("leaf mask d", leaf_mask_d),
        # ("leaf mask z", leaf_mask_z),
        ("Leaf Masked", masked),
        # ("roi", roi),
        # ("filtered mask", filtered_mask),
        ("Size Analysis", size_analysis_image),
    ]

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

    for image_path, class_index in tqdm(dataset):
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
        transformed_images = transform_image(args.filename)
        display_images("Transformation", transformed_images)
    elif args.src and args.dst and not args.filename:
        print(f"Reading from source: {args.src} and write to destination: {args.dst}")
        transform_dataset(Path(args.src), Path(args.dst))
    else:
        parser.error("You must provide either a filename or both -src and -dst options.")
        parser.print_help()
