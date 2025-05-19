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

def apply_gaussian(img_path):
    image = cv.imread(img_path)
    gaussian_3_3 = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    output = cv.filter2D(image, -1, gaussian_3_3)
    # gaussian_5_5 = (1 / 159) * np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])
    # output = cv2.filter2D(image, -1, gaussian_5_5)
    return output


def apply_vertical(img):
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = cv.filter2D(img, -1, sobel_x_kernel)
    return output

def apply_horizontal(img):
    sobel_x_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = cv.filter2D(img, -1, sobel_x_kernel)
    return output

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
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    # choose a channel that correctly separate background and the leaf
    # grayscale_img = pcv.rgb2gray_hsv(rgb_img=img, channel="s")

    grayscale_img = pcv.rgb2gray_lab(rgb_img=numpy_array, channel="a")
    # plot the histogram of the grayscale values
    # hist = pcv.visualize.histogram(img=grayscale_img, bins=30)
    # plt.figure()
    # plt.hist(grayscale_img.ravel(), bins=30, range=[0, 255])
    # plt.title('grayscale image histogram')
    # plt.xlabel('pixel intensity')
    # plt.ylabel('Frequency')
    # plt.show()
    # set_thresh = pcv.threshold.binary(gray_img=grayscale_img, threshold=80, object_type="light")
    otsu = pcv.threshold.otsu(gray_img=grayscale_img, object_type="light")
    blury = pcv.median_blur(gray_img=otsu, ksize=5)
    # auto_thresh, _ = pcv.threshold.custom_range(img=img, lower_thresh=[40, 50, 0], upper_thresh=[75, 95, 35], channel="rgb")
    fill_image = pcv.fill_holes(bin_img=otsu)
    roi = pcv.roi.circle(fill_image, x=125, y=125, r=100)
    kept_mask = pcv.roi.filter(mask=fill_image, roi=roi, roi_type="partial")
    analysis_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    # analysis_color = pcv.analyze.color(rgb_img=img, labeled_mask=kept_mask, colorspaces="rgb")

    return s, b, a, grayscale_img, otsu, blury, fill_image, roi, kept_mask

def segmenting_plant(numpy_array):
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

def segmenting_background(numpy_array):
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

def transform_image(image_path, show=False):
    pil_image = Image.open(image_path)
    # pil_image.show()

    numpy_image = np.asarray(pil_image)
    cv_image = cv.imread(image_path)

    # make a tuple of image with their name
    # gaussian_image = gaussian_blur(numpy_image)
    # canny_image = canny_edge_detection(gaussian_image)
    # canny_image = canny_edge_detection(numpy_image)
    # s, b, a, grayscale_img, otsu, blury, fill_image, roi, kept_mask = segmentation(numpy_image, cv_image)
    a_channel, median_img, otsu, gaussian_img, leaf_mask_a, masked = segmenting_plant(numpy_image)
    b_channel, median_img, otsu, gaussian_img, leaf_mask_b, masked = segmenting_background(numpy_image)
    s_channel, median_img, otsu, gaussian_img, leaf_mask_s, masked = segmenting_saturation(numpy_image)

    leaf_mask_c = pcv.logical_or(bin_img1=leaf_mask_a, bin_img2=leaf_mask_b)
    leaf_mask_d = pcv.logical_or(bin_img1=leaf_mask_s, bin_img2=leaf_mask_b)
    leaf_mask_z = pcv.logical_or(bin_img1=leaf_mask_c, bin_img2=leaf_mask_d)
    # masked = pcv.apply_mask(img=numpy_image, mask=leaf_mask_c, mask_color='white')
    masked = pcv.apply_mask(img=numpy_image, mask=leaf_mask_z, mask_color='white')
    # roi = pcv.roi.circle(img=masked, x=125, y=125, r=100)
    # filtered_mask = pcv.roi.filter(mask=leaf_mask_z, roi=roi, roi_type='partial')
    analysis_image = pcv.analyze.size(img=numpy_image, labeled_mask=leaf_mask_z)

    transformed_images = [
        ("original: pil, numpy", numpy_image),
        ("s channel", s_channel),
        ("b channel", b_channel),
        ("a channel", a_channel),
        # ("median blur", median_img),
        # ("canny_edge_detection", canny_image),
        # ("otsu", otsu),
        # ("gaussian_img", gaussian_img),
        # ("leaf mask b", leaf_mask_b),
        # ("leaf mask a", leaf_mask_a),
        ("leaf mask c", leaf_mask_c),
        ("leaf mask d", leaf_mask_d),
        ("leaf mask z", leaf_mask_z),
        ("leaf masked", masked),
        # ("roi", roi),
        # ("filtered mask", filtered_mask),
        ("analysis image", analysis_image),
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
