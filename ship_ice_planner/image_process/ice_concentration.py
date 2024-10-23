""" Compute sea ice concentration from a binary image """
from typing import Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.signal import fftconvolve


def compute_sic(bin_img: np.ndarray,
                kernel: Tuple,
                stride=1,
                sic_size=None,
                show_plot=False,
                edges_ice=None):
    """
    Compute sea ice concentration from a binary image of sea ice. The sea ice concentration (SIC) is computed by
    convolving the binary image with a kernel of size `kernel` and stride `stride`. The SIC is computed as the ratio
    of the number of ice pixels to the total number of pixels in the kernel.

    :param bin_img: binary image of sea ice
    :param kernel: kernel size for the 2D image convolution
    :param stride: stride for the 2D image convolution
    :param sic_size: size of the SIC image to return
    :param show_plot: whether to show a plot of the SIC computation
    :param edges_ice: for plotting purposes, the segmented ice edges
    """
    assert kernel[0] % 2 == 1 and kernel[1] % 2 == 1, \
        'Kernel size must be odd, got kernel size {}'.format(kernel)
    # if stride is not None:
    #     assert bin_img.shape[0] % stride == 0 and bin_img.shape[1] % stride == 0, \
    #         'Stride must divide image size, got stride {} and image size {}'.format(stride, bin_img.shape)

    # add padding
    padded_bin_img = cv2.copyMakeBorder(bin_img.astype('float32'),
                                        kernel[0] // 2, kernel[0] // 2, kernel[1] // 2, kernel[1] // 2,
                                        cv2.BORDER_REFLECT_101)
    sic = fftconvolve(padded_bin_img, np.ones(kernel) / np.prod(kernel), mode='valid')[::stride, ::stride]

    if sic_size is None:
        sic_size = (bin_img.shape[1], bin_img.shape[0])
    if sic_size[::-1] != sic.shape:
        sic = cv2.resize(sic, dsize=sic_size, interpolation=cv2.INTER_NEAREST_EXACT)

    # ensure SIC is between 0 and 1
    sic = cv2.min(1, cv2.max(0, sic))

    if show_plot:
        f, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ax[0].imshow(bin_img, cmap='gray', origin='lower')
        ax[1].imshow(sic, cmap='gray', origin='lower')
        # to show the size of the kernel or the matrix that is being convolved
        rect = patches.Rectangle((0, bin_img.shape[0] / 2), kernel[1], kernel[0],
                                 linewidth=1, edgecolor='r', facecolor='none')
        # to show the stride
        rect2 = patches.Rectangle((stride, bin_img.shape[0] / 2 + stride), kernel[1], kernel[0],
                                  linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        ax[0].add_patch(rect2)
        if edges_ice is not None:
            edges_ice_im = np.zeros(edges_ice.shape + (4,))  # init RGBA array
            edges_ice_im[:] = colors.to_rgba('m')
            edges_ice_im[:, :, 3] = edges_ice[::-1]  # set pixel transparency to 0 if pixel value is 0
            ax[1].imshow(edges_ice_im, extent=(0, sic.shape[1], sic.shape[0], 0), origin='lower')
        plt.show()

    return sic
