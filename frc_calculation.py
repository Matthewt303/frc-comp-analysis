#!/home/kxtz813/.conda/envs/matthew_dev/bin/python3

import numpy as np
from scipy.fft import fft2
from scipy.signal.windows import tukey
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from numba import jit
import cupy as cp
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

def bin_localizations(locs: 'np.ndarray', size:int=None,
                      mag:float=1) -> 'np.ndarray':
    
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
        
    image - super-resolution image
    
    """

    locs = locs.copy()

    locs = (locs + 0.5) * mag

    x_locs, y_locs = locs[:, 0], locs[:, 1]

    if size is None:

        size = np.int32(np.ceil(np.max(locs)))

    locs_to_keep = (
        (x_locs >= 0) & (x_locs < size) & 
        (y_locs >= 0) & (y_locs < size)
        )

    new_locs = locs[locs_to_keep]

    image = np.zeros((size, size), dtype=np.float32)
    
    for x, y in new_locs:

        im_x, im_y = np.int32(x), np.int32(y)

        if 0 <= im_x < size and 0 <= im_y < size:

            image[im_x, im_y] += 1

    return image

def apply_tukey_window(image: 'np.ndarray', alpha:float=0.5) -> 'np.ndarray':
    
    """
    Summary:
    
    Tapers the edges of an image to prevent spectral leakage after FT
    --------------------------------
    Inputs:
    
    image - the image
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

def radial_count(image: 'np.ndarray') -> np.ndarray:
    
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

def calculate_sigma_curve(counts: 'np.ndarray', sigma:int=3) -> 'np.ndarray':
    
    curve = sigma / np.sqrt(counts / 2)
    
    return curve

def calc_three_sigma(image: 'np.ndarray') -> 'np.ndarray':
    
    radius_counts = radial_count(image)
    
    three_sigma = calculate_sigma_curve(radius_counts)
    
    return three_sigma

@jit(nopython=True, nogil=True, cache=False)
def radial_sum_numba(image: 'np.ndarray') -> np.ndarray:
    
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

def cpu_fft(image: 'np.ndarray') -> 'np.ndarray':
    
    """
    Summary:
    
    Calculates the 2D fast fourier transform of an image.
    --------------------------------
    Inputs:
    
    image - 2D array of image data
    --------------------------------
    Output:
    
    Fourier transformed image
    
    """
    
    im = image.copy()
    
    return fft2(im, workers=8).astype(np.complex64)


def frc(im1: 'np.ndarray', im2: 'np.ndarray') -> 'np.ndarray':
    
    edge = int(im1.shape[0] / 2)
    
    im1_tukey = apply_tukey_window(im1)
    im2_tukey = apply_tukey_window(im2)
    
    print('starting FFT')
    start = time.time()
    
    im1_ft = np.fft.fftshift(cpu_fft(im1_tukey)).astype(np.complex64)
    im2_ft = np.fft.fftshift(cpu_fft(im2_tukey)).astype(np.complex64)
    
    end = time.time()
    print('FFT ended')
    print('FFT took ' + str(end - start) + ' seconds')
    
    print('Starting FRC calc')
    start2 = time.time()
    
    num = np.real(radial_sum_numba(im1_ft * np.conj(im2_ft)))
    denom1 = np.sqrt(radial_sum_numba(np.abs(im1_ft) ** 2)) 
    denom2 = np.sqrt(radial_sum_numba(np.abs(im2_ft) ** 2))
    
    denom = denom1 * denom2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        
        frc_curve = num / denom
        frc_curve[np.isnan(frc_curve)] = 0
    
    end2 = time.time()
    print('FRC calc done')
    print('FRC calc took ' + str(end2 - start2) + ' seconds')
    
    return frc_curve[0:edge]

def split_half_simple(localisations: 'np.ndarray'):
    
    halfway_point = localisations.shape[0] // 2
    
    set1 = localisations[0:halfway_point, 1:]
    set2 = localisations[halfway_point:, 1:]
    
    return set1, set2

def split_odd_even(localisations: 'np.ndarray'):
    
    odd_indices = np.where(localisations[:, 0] % 2 == 1)
    even_indices = np.where(localisations[:, 0] % 2 == 0)
    
    set1, set2 = localisations[odd_indices], localisations[even_indices]
    set1, set2 = set1[:, 1:], set2[:, 1:]
    
    return set1, set2

def calculate_frc(localisations: 'np.ndarray', frames,
                  magnification: float, size=None) -> 'np.ndarray':
    
    if size is None:
        
        xy = magnification * localisations[:, 1:]  
        
        size = np.int64(np.ceil(np.max(xy))) + 100
        
    if frames == 'n':
    
        set1, set2 = split_half_simple(localisations)
    
    elif frames == 'y':
        
        set1, set2 = split_odd_even(localisations)
    
    image1 = bin_localizations(set1, size=size, mag=magnification)
    image2 = bin_localizations(set2, size=size, mag=magnification)
    print('images binned')
    
    frc_vals = frc(image1, image2)
    x_pix = np.arange(len(frc_vals)) / image1.shape[0]
    x_spatial_frequency = x_pix * magnification
    
    return frc_vals, x_spatial_frequency

def calculate_frc_sigma(localisations: 'np.ndarray', frames,
                  magnification: float, size=None) -> 'np.ndarray':
    
    if size is None:
        
        xy = magnification * localisations[:, 1:]  
        
        size = np.int64(np.ceil(np.max(xy))) + 100
        
    if frames == 'n':
    
        set1, set2 = split_half_simple(localisations)
    
    elif frames == 'y':
        
        set1, set2 = split_odd_even(localisations)
    
    image1 = bin_localizations(set1, size=size, mag=magnification)
    image2 = bin_localizations(set2, size=size, mag=magnification)
    print('images binned')
    
    three_sigma_curve = calc_three_sigma(image1)
    
    frc_vals = frc(image1, image2)
    x_pix = np.arange(len(frc_vals)) / image1.shape[0]
    x_spatial_frequency = x_pix * magnification
    
    return frc_vals, x_spatial_frequency, three_sigma_curve

def smooth_frc(frc: 'np.ndarray', spatial_frequency: 'np.ndarray') -> 'np.ndarray':
    
    lowess = sm.nonparametric.lowess
    
    smoothed_data = lowess(frc, spatial_frequency, frac=0.025)
    
    return smoothed_data

def smooth_savgol(frc: 'np.ndarray') -> 'np.ndarray':
    
    frc_curve = frc.copy()
    
    interval = frc_curve.shape[0] // 20
    
    frc_smooth = savgol_filter(frc_curve, interval, 2)
    
    return frc_smooth

def temp_plot(frc, v, thold):
    
    mpl.rcParams['font.sans-serif'] = ['Nimbus Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28
    
    threshold_plot = np.full(frc.shape[0], thold)
    
    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)
    
    ax.plot(v, frc, 'darkmagenta', label='FRC', linewidth=4.5)
    ax.plot(v, threshold_plot, 'royalblue', label='Threshold', linewidth=4.5)
    
    leg = plt.legend(loc='upper right')
    
    for line in leg.get_lines():
        line.set_linewidth(3.5)
    
    for text in leg.get_texts():
        text.set_fontsize(28)
    
    ax.set_xlim(left=0)
    
    ratio = 1.0
    
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    
    ax.tick_params(axis='y', which='major', length=6, direction='in', pad=10)
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in', pad=10)
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.set_xlabel(r'Spatial frequency $\mathregular{(nm^{-1})}$', labelpad=-3, fontsize=36)
    ax.set_ylabel('Fourier Ring Correlation', labelpad=2, fontsize=36)
    
    plt.savefig('/home/kxtz813/test/frc_plot.svg')

def calculate_frc_resolution(frc: 'np.ndarray', spatial_frequency: 'np.ndarray',
                             threshold=0.143) -> float:
    
    temp_plot(frc, spatial_frequency, threshold)
    
    intercept = np.argwhere(np.diff(np.sign(frc - threshold))).flatten()
    
    resolution_spat_freq = spatial_frequency[intercept][0]
    
    intercept_y_coord = frc[intercept][0]
    
    return resolution_spat_freq, intercept_y_coord

def calculate_frc_res_sigma(frc: 'np.ndarray', spatial_frequency: 'np.ndarray',
                            sigma_curve: 'np.ndarray') -> float:
    
    #curve_below_one = sigma_curve[sigma_curve < 0.95]
    
    #curve_below_one_lowhigh = np.sort(curve_below_one)
    
    #curve_below_one_highlow = curve_below_one_lowhigh[::-1]
    
    #trunc_frc = frc[:int(curve_below_one_highlow.shape[0])]
    
    intercepts = np.argwhere(np.diff(np.sign(frc - sigma_curve))).flatten()
    
    resolution_spat_freq = spatial_frequency[intercepts][0]
    
    intercept_y_coord = frc[intercepts][0]
    
    return resolution_spat_freq, intercept_y_coord

def frc_fixed(locs: 'np.ndarray', magnification: float, frames_used: str):
    
    frc_curve, spatial_freqs = calculate_frc(locs, frames_used, magnification)
    
    frc_smoothed = smooth_savgol(frc_curve)

    frc_res, frc_res_val = calculate_frc_resolution(frc_smoothed, spatial_freqs)
    
    return frc_smoothed, spatial_freqs, 1 / frc_res, frc_res_val

def frc_sigma(locs: 'np.ndarray', magnification: float, frames_used: str):
    
    frc_curve, spatial_freqs, radii = calculate_frc_sigma(locs, frames_used, magnification)
    
    frc_smoothed = smooth_savgol(frc_curve)

    frc_res, frc_res_val = calculate_frc_res_sigma(frc_smoothed, spatial_freqs, radii)
    
    return frc_smoothed, spatial_freqs, 1 / frc_res, frc_res_val, radii

def calculate_p_value(raw_frc: 'np.ndarray', denoised_frc: 'np.ndarray') -> float:
    
    test_result = mannwhitneyu(raw_frc, denoised_frc)
    p_value = test_result[1]
    
    return p_value

