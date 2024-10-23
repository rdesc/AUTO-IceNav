""" Runs the watershed segmentation algorithm to segment ice """
import argparse
import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt, colors
from tqdm import tqdm


__all__ = [
    'segment_ice',
    'get_ice_binary_occgrid',
    'get_ice_edges',
]


def segment_ice(im, dt_threshold=0.2, show_plot=False):
    """
    Segment ice using watershed algorithm https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

    :param im: image to segment
    :param dt_threshold: threshold for thresholding distance transformed image
    :param show_plot: whether to show the plot
    """
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(src=opening, distanceType=cv2.DIST_L2, maskSize=5)
    ret, sure_fg = cv2.threshold(dist_transform, dt_threshold * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    if show_plot:
        img[markers == -1] = [255, 0, 0]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        plt.imshow(markers)
        plt.show()

    return markers


def get_ice_binary_occgrid(segmented_image):
    """
    Returns a binary image of the segmented regions
    Where 1 indicates ice and 0 otherwise
    """
    binary_occgrid = segmented_image.copy()
    binary_occgrid[binary_occgrid != 1] = 0
    return ~binary_occgrid.astype(bool)


def get_ice_edges(segmented_image):
    """
    Returns a binary image of the edges of the segmented regions
    Where 1 indicates an ice edge and 0 otherwise
    """
    edges = segmented_image.copy()
    edges[edges != -1] = 0
    edges[edges == -1] = 1
    return edges.astype(bool)


def demo(image_directory, animate=False):
    from ship_ice_planner.image_process.ice_concentration import compute_sic

    files = glob.glob(os.path.join(image_directory, '*.png'))
    files = sorted(files, key=lambda x: int(os.path.basename(x).split(sep='-')[-1].split('.')[0]))

    if animate:
        plt.figure(figsize=(6, 6))

    for idx, fp in tqdm(enumerate(files)):
        print(fp)
        img = cv2.imread(fp)

        segmented_image = segment_ice(img, show_plot=not animate)
        binary_ice_im = get_ice_binary_occgrid(segmented_image)
        edges_ice = get_ice_edges(segmented_image)

        sea_ice_concentration = compute_sic(binary_ice_im,
                                            kernel=(201, 201),
                                            stride=40,
                                            show_plot=not animate,
                                            edges_ice=edges_ice,
                                            )

        if animate:
            plt.cla()
            plt.imshow(sea_ice_concentration, cmap='gray')
            edges_ice_im = np.zeros(edges_ice.shape + (4,))  # init RGBA array
            edges_ice_im[:] = colors.to_rgba('m')
            edges_ice_im[:, :, 3] = edges_ice  # set pixel transparency to 0 if pixel value is 0
            plt.imshow(edges_ice_im, vmax=1, vmin=0, extent=(0, sea_ice_concentration.shape[1],
                                                             sea_ice_concentration.shape[0], 0))
            plt.pause(0.01)  # animate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_directory', type=str, help='path to folder containing images,'
                                                          ' e.g. ~/Downloads/best_lattice_frames')
    parser.add_argument('--animate', action='store_true', help='whether to animate the SIC computation')
    args = parser.parse_args()

    demo(args.image_directory, args.animate)
