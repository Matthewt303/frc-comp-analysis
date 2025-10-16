#!/home/kxtz813/.conda/envs/matthew_dev/bin/python3

import numpy as np
from scipy.fft import fft2
from scipy.signal.windows import tukey
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
from numba import jit
import time
import os


@jit(nopython=True, nogil=True, cache=False)
def bin_image(locs: "np.ndarray", size: int) -> "np.ndarray":
    """
    Summary:

    Bins localisation data into an image.
    --------------------------------
    Inputs:

    locs - xy localisation data
    size - Maximum dimensions (in pixels) of the image, determined by max xy position
    --------------------------------
    Output:
    image - super-resolution image
    """

    image = np.zeros((size, size), dtype=np.float32)

    for x, y in locs:
        im_x, im_y = np.int32(x), np.int32(y)

        if 0 <= im_x < size and 0 <= im_y < size:
            image[im_x, im_y] += 1

    return image


def bin_localizations(
    locs: "np.ndarray", size: int = None, mag: float = 1
) -> "np.ndarray":
    """
    Summary:

    Converts xy localisation data into an image with a user-specified
    scaling factor
    --------------------------------
    Inputs:

    locs - xy localisation data
    size - Maximum dimensions (in pixels) of the image, determined by max xy position
    mag - scaling of image. E.g. for locs in nm, mag = 0.1 = 0.1 pix/nm = 10 nm / pix
    --------------------------------
    Output:

    image - super-resolution image as a 2D array with dimensions of size x size.

    """

    locs = locs.view()

    locs = (locs + 0.5) * mag

    x_locs, y_locs = locs[:, 0], locs[:, 1]

    if size is None:
        size = np.int64(np.ceil(np.max(locs)))

    locs_to_keep = (x_locs >= 0) & (x_locs < size) & (y_locs >= 0) & (y_locs < size)

    new_locs = locs[locs_to_keep]

    image = bin_image(new_locs, size)

    return image


def apply_tukey_window(image: "np.ndarray", alpha: float = 0.5) -> "np.ndarray":
    """
    Summary:

    Tapers the edges of an image to prevent spectral leakage after FT
    --------------------------------
    Inputs:

    image - the image as a 2D array with dimensions N x N
    alpha - specifies the window over which to taper. Smaller alpha implies
    a smaller tapering window. Default value = 0.5
    --------------------------------
    Output:

    filt_image - image with edges tapered

    """

    win_x = tukey(image.shape[0], alpha)
    win_y = tukey(image.shape[1], alpha)

    window = np.outer(win_x, win_y)

    filt_image = image * window

    return filt_image


@jit(nopython=True, nogil=True, cache=False)
def radial_count(image: "np.ndarray") -> np.ndarray:
    """
    Summary:

    Calculates the number of pixels bounded by the circles used for Fourier ring
    correlation. The radii are calculated by first acquiring the indices of the image
    and subtracting the center indices, resulting in an array centered at 0, 0 with
    each pixel denoting the radius from the center.
    --------------------------------
    Inputs:

    image - the image as an N x N 2D array.
    --------------------------------
    Output:

    count - the number of pixels bound by each radius value.

    """
    edge = int(image.shape[0] / 2)

    y, x = np.indices(image.shape)

    center = np.array(image.shape) // 2

    r = np.hypot(x - center[1], y - center[0])
    r_int = r.astype(np.int32)
    max_r = r_int.max() + 1

    count = np.zeros(max_r, dtype=np.int32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ri = r_int[i, j]
            count[ri] += 1

    return count.astype(np.float32)[0:edge]


def calculate_sigma_curve(counts: "np.ndarray", sigma: int = 3) -> "np.ndarray":
    """
    Summary:

    Calculates the 3-sigma curve for resolution evaluation.
    --------------------------------
    Inputs:

    counts - 1D array containing the number of pixels bound by a circle of radius, q,
    for each radius
    --------------------------------
    Output:

    curve - 1D array of the three-sigma curve.

    """

    curve = sigma / np.sqrt(counts / 2)

    return curve


def calc_three_sigma(image: "np.ndarray") -> "np.ndarray":
    """
    Summary:

    Function that combines radial_count and calculate_three_sigma_curve.
    --------------------------------
    Inputs:

    image - the image as an N x N 2D array.
    --------------------------------
    Output:

    three_sigma - 1D array of the three-sigma curve.

    """
    radius_counts = radial_count(image)

    three_sigma = calculate_sigma_curve(radius_counts)

    return three_sigma


@jit(nopython=True, nogil=True, cache=False)
def radial_sum_numba(image: "np.ndarray") -> np.ndarray:
    """
    Summary:

    Calculates the radial sum starting from the center of an image. I.e.
    adds up the intensities of image pixels within varying radii.
    Each count is normalized against the counts of the radius.
    --------------------------------
    Inputs:

    image - 2D array of image data
    --------------------------------
    Output:

    radial_profile - 1D array with total intensity at each radius

    """

    y, x = np.indices(image.shape)

    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2

    # Calculate the radial distances
    r = np.hypot(x - center_x, y - center_y)
    r_int = r.astype(np.int32)
    max_r = r_int.max() + 1

    # Initialize the output array
    radial_profile = np.zeros(max_r, dtype=np.float32)
    count = np.zeros(max_r, dtype=np.int32)

    # Calculate the radial sum
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ri = r_int[i, j]
            radial_profile[ri] += np.real(image[i, j])
            count[ri] += 1

    # Normalize by the count of each radius
    for i in range(max_r):
        if count[i] > 0:
            radial_profile[i] /= count[i]

    return radial_profile


def cpu_fft(image: "np.ndarray") -> "np.ndarray":
    """
    Summary:

    Calculates the 2D fast fourier transform of an image. The function uses half
    of the available CPUs for the calculation, as specified by the 'workers'
    keyword argument.
    --------------------------------
    Inputs:

    image - 2D array of image data
    --------------------------------
    Output:

    Fourier transformed image

    """

    im = image.view()

    return fft2(im, workers=os.cpu_count() // 2).astype(np.complex64)


def frc(im1: "np.ndarray", im2: "np.ndarray") -> "np.ndarray":
    """
    Summary:

    Calculates the Fourier ring correlation curve from two half-images. The function
    is composed of: apply_tukey_window, cpu_fft, and radial_sum_numba.
    --------------------------------
    Inputs:

    im1 - 2D array of one half-image
    im2 - 2D array of the second half-image
    --------------------------------
    Output:

    frc_curve - Fourier ring correlation curve as 1D array.

    """

    edge = int(im1.shape[0] / 2)

    im1_tukey = apply_tukey_window(im1)
    im2_tukey = apply_tukey_window(im2)

    print("starting FFT \n")
    start = time.time()

    im1_ft = np.fft.fftshift(cpu_fft(im1_tukey)).astype(np.complex64)
    im2_ft = np.fft.fftshift(cpu_fft(im2_tukey)).astype(np.complex64)

    end = time.time()
    print("FFT ended \n")
    print("FFT took " + str(end - start) + " seconds \n")

    print("Starting FRC calc \n")
    start2 = time.time()

    num = np.real(radial_sum_numba(im1_ft * np.conj(im2_ft)))
    denom1 = np.sqrt(radial_sum_numba(np.abs(im1_ft) ** 2))
    denom2 = np.sqrt(radial_sum_numba(np.abs(im2_ft) ** 2))

    denom = denom1 * denom2

    with np.errstate(divide="ignore", invalid="ignore"):
        frc_curve = num / denom
        frc_curve[np.isnan(frc_curve)] = 0

    end2 = time.time()
    print("FRC calc done \n")
    print("FRC calc took " + str(end2 - start2) + " seconds \n")

    return frc_curve[0:edge]


def split_half_simple(localisations: "np.ndarray"):
    halfway_point = localisations.shape[0] // 2

    set1 = localisations[0:halfway_point, 1:]
    set2 = localisations[halfway_point:, 1:]

    return set1, set2


def split_odd_even(localisations: "np.ndarray"):
    odd_indices = np.where(localisations[:, 0] % 2 == 1)
    even_indices = np.where(localisations[:, 0] % 2 == 0)

    set1, set2 = localisations[odd_indices], localisations[even_indices]
    set1, set2 = set1[:, 1:], set2[:, 1:]

    return set1, set2


def calculate_frc(
    localisations: "np.ndarray", split_method: str, magnification: float, size=None
) -> "np.ndarray":
    if size is None:
        xy = magnification * localisations[:, 1:]

        size = np.int64(np.ceil(np.max(xy))) + 100

    if split_method == "simple":
        set1, set2 = split_half_simple(localisations)

    elif split_method == "odd_even":
        set1, set2 = split_odd_even(localisations)

    print("starting image binning \n")
    start = time.time()
    image1 = bin_localizations(set1, size=size, mag=magnification)
    image2 = bin_localizations(set2, size=size, mag=magnification)
    print("images binned")
    end = time.time()
    print(str(end - start) + " seconds \n")

    frc_vals = frc(image1, image2)
    x_pix = np.arange(len(frc_vals)) / image1.shape[0]
    x_spatial_frequency = x_pix * magnification

    return frc_vals, x_spatial_frequency


def calculate_frc_sigma(
    localisations: "np.ndarray", split_method: str, magnification: float, size=None
) -> "np.ndarray":
    if size is None:
        xy = magnification * localisations[:, 1:]

        size = np.int64(np.ceil(np.max(xy))) + 100

    if split_method == "simple":
        set1, set2 = split_half_simple(localisations)

    elif split_method == "odd_even":
        set1, set2 = split_odd_even(localisations)

    image1 = bin_localizations(set1, size=size, mag=magnification)
    image2 = bin_localizations(set2, size=size, mag=magnification)
    print("images binned")

    three_sigma_curve = calc_three_sigma(image1)

    frc_vals = frc(image1, image2)
    x_pix = np.arange(len(frc_vals)) / image1.shape[0]
    x_spatial_frequency = x_pix * magnification

    return frc_vals, x_spatial_frequency, three_sigma_curve


def smooth_savgol(frc: "np.ndarray") -> "np.ndarray":
    frc_curve = frc.view()

    interval = frc_curve.shape[0] // 20

    frc_smooth = savgol_filter(frc_curve, interval, 2)

    return frc_smooth


def calculate_frc_resolution(
    frc: "np.ndarray", spatial_frequency: "np.ndarray", threshold=0.143
) -> float:
    intercept = np.argwhere(np.diff(np.sign(frc - threshold))).flatten()

    resolution_spat_freq = spatial_frequency[intercept][0]

    intercept_y_coord = frc[intercept][0]

    return resolution_spat_freq, intercept_y_coord


def calculate_frc_res_sigma(
    frc: "np.ndarray", spatial_frequency: "np.ndarray", sigma_curve: "np.ndarray"
) -> float:
    curve_below_one = sigma_curve[sigma_curve < 0.95]

    curve_below_one_lowhigh = np.sort(curve_below_one)

    curve_below_one_highlow = curve_below_one_lowhigh[::-1]

    trunc_frc = frc[: int(curve_below_one_highlow.shape[0])]

    intercepts = np.argwhere(
        np.diff(np.sign(trunc_frc - curve_below_one_highlow))
    ).flatten()

    resolution_spat_freq = spatial_frequency[intercepts][0]

    intercept_y_coord = frc[intercepts][0]

    return resolution_spat_freq, intercept_y_coord


def frc_fixed(locs: "np.ndarray", magnification: float, split_method: str):
    frc_curve, spatial_freqs = calculate_frc(locs, split_method, magnification)

    frc_smoothed = smooth_savgol(frc_curve)

    frc_res, frc_res_val = calculate_frc_resolution(frc_smoothed, spatial_freqs)

    return frc_smoothed, spatial_freqs, 1 / frc_res, frc_res_val


def frc_sigma(locs: "np.ndarray", magnification: float, split_method: str):
    frc_curve, spatial_freqs, sig_curve = calculate_frc_sigma(
        locs, split_method, magnification
    )

    frc_smoothed = smooth_savgol(frc_curve)

    frc_res, frc_res_val = calculate_frc_res_sigma(
        frc_smoothed, spatial_freqs, sig_curve
    )

    return frc_smoothed, spatial_freqs, 1 / frc_res, frc_res_val, sig_curve


def calculate_p_value(raw_frc: "np.ndarray", denoised_frc: "np.ndarray") -> float:
    test_result = mannwhitneyu(raw_frc, denoised_frc)
    p_value = test_result[1]

    return p_value
