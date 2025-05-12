# related to this paper:
# https://doi.org/10.1051/itmconf/20224403002

# I need to extract this features in Transformation.py
# area, perimeter, red_mean, green_mean, blue_mean
# red_std, green_std, blue_std, f1, f2, f4, f5, f6, f7, f8
import argparse
from plantcv import plantcv as pcv
from utils import DatasetFolder, Dataloader
import cv2
import numpy as np
# from rembg import remove


def show_cv2_img(img, name=None):
    if isinstance(img, str):
        img = cv2.imread(img)
        name = str(img)
    # if not name:
    #     name = str(type(img))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def apply_gaussian(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gaussian_kernel = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    output = cv2.filter2D(image, -1, gaussian_kernel)
    return output


def apply_vertical(img):
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = cv2.filter2D(img, -1, sobel_x_kernel)
    return output

def apply_horizontal(img):
    sobel_x_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = cv2.filter2D(img, -1, sobel_x_kernel)
    return output


# def remove_bg(img_path):
#     img = cv2.imread(img_path)
#     output = remove(img)
#     return output


def segment_leaf(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green color threshold (adjust for specific leaves)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours
    result = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

    return result


def segmentation(img_path):
    img, img_path, img_filename = pcv.readimage(img_path)
    # print(f"image: {img}")
    # print(f"image filename: {img_filename}")
    # print(f"image directory: {img_path}")
    # show_cv2_img(img)

    # gray scale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show_cv2_img(gray_image)

    # gaussian filter
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    # show_cv2_img(blurred_image)

    # otsu
    masked, otsu_thresholded = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # th3 = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    # th2 = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -2)
    # show_cv2_img(masked, "masked")
    show_cv2_img(otsu_thresholded, "otsu_thresholded")
    # show_cv2_img(th3, "th3")
    # show_cv2_img(th2, "th2")

    # morphological Transformation
    kernel = np.ones((3, 3), np.uint8)
    closed_image = cv2.morphologyEx(otsu_thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    show_cv2_img(closed_image, "closed_image")

    # Bitwise AND Operation
    # show_cv2_img(masked)
    # inverted_mask = cv2.bitwise_not(closed_image)
    # show_cv2_img(inverted_mask)
    # mask = closed_image[:, :, np.newaxis]  # Add a channel dimension to the binary image
    # segmented_image = cv2.bitwise_and(img, img, mask=mask)
    segmented_image = cv2.bitwise_and(img, img, mask=closed_image)
    # segmented_image = cv2.bitwise_and(img, img, mask=otsu_thresholded)
    # segmented_path = f"{img_path}{img_filename}_segmented.jpg"
    # pcv.print_image(segmented_image, segmented_path)
    show_cv2_img(segmented_image, "segmented_image")

    # PlantCV Transformation

    colorspace_img = pcv.visualize.colorspaces(rgb_img=img, original_img=False)
    grayscale_img = pcv.rgb2gray_hsv(rgb_img=img, channel="s")

    plantcv_gaussian_blur = pcv.gaussian_blur(img, (5, 5), 0)
    plantcv_gaussian_blur = pcv.gaussian_blur(grayscale_img, (5, 5), 0)

    pcv_edge = pcv.canny_edge_detect(grayscale_img, sigma=2)

    auto_thresh = pcv.threshold.otsu(gray_img=grayscale_img, object_type="light")
    fill_image = pcv.fill_holes(bin_img=auto_thresh)
    roi = pcv.roi.circle(fill_image, x=125, y=125, r=100)
    kept_mask = pcv.roi.filter(mask=fill_image, roi=roi, roi_type="partial")
    # analysis_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)
    # analysis_color = pcv.analyze.color(rgb_img=img, labeled_mask=kept_mask, colorspaces="rgb")

    # morphological Transformation
    segmented_image = cv2.bitwise_and(img, img, mask=fill_image)
    # show_cv2_img(segmented_image)
    # segmented_path = f"{img_path}{img_filename}_segmented.jpg"
    # pcv.print_image(segmented_image, segmented_path)
    # analysis_image = pcv.analyze.size(img=segmented_image, labeled_mask=kept_mask)
    # analysis_color = pcv.analyze.color(rgb_img=segmented_image, labeled_mask=kept_mask, colorspaces="rgb")

    return segmented_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the random forest algorithm")
    parser.add_argument("dataset", help="dataset path")
    args = parser.parse_args()

    pcv.params.debug = "plot"
    dataset = DatasetFolder(args.dataset)

    dataloader = Dataloader(dataset, batch_size=1, shuffle=True)

    one_batch = next(iter(dataloader))
    for img_path, class_index in one_batch:
        # img_segmented = segmentation(img_path)
        # show_cv2_img(img_segmented, "img_segmented")

        # img_without = remove_bg(img_path)
        # show_cv2_img(img_without, "rembg image")

        # img_leaf = segment_leaf(img_path)
        # show_cv2_img(img_leaf, "img_leaf segmentation")

        # img, mask = segmentation(img_path)
        # pcv.analyse.colorspaces()

        gaussian = apply_gaussian(img_path)
        show_cv2_img(gaussian, "gaussian image")
        vertical = apply_vertical(gaussian)
        show_cv2_img(vertical, "vertical image")
        horizontal = apply_horizontal(gaussian)
        show_cv2_img(horizontal, "horizontal image")
