#!/home/kxtz813/.conda/envs/matthew_dev/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 10:00:28 2025

@author: kxtz813
"""

import numpy as np
from scipy.fft import fft2
import pyfftw
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

def scipy_fft():
    
    image = 1000 * np.random.rand(3781, 3781).astype(np.float32)
    
    power_of_two = int(np.ceil(np.log2(image.shape[0])))
    
    if (2 ** power_of_two - image.shape[0]) % 2 == 1:
        
        image = np.pad(image, (0, 1), mode='constant', constant_values=0)
    
    w = int((2 ** power_of_two - image.shape[0]) / 2)
    
    image = np.pad(image, (w, w), mode='constant', constant_values=0)
    
    return fft2(image)

def og_scipy_fft():
    
    image = 1000 * np.random.rand(3781, 3781).astype(np.float32)
    
    return fft2(image, workers=8)


def fast_fft():
    
    image = np.random.rand(1024, 1024) * 1000
    
    fft2 = pyfftw.builders.fft2(image)
    
    fft_result = fft2()
    
    return fft_result

def gpu_fft():
    
    image = np.random.rand(1024, 1024) * 1000
    image_gpu = cp.asarray(image)
    fft_result_gpu = cp.fft.fft2(image_gpu)
    
    return cp.asnumpy(fft_result_gpu)

def numpy_load():
    
    path = '/home/kxtz813/test/test.csv'
    
    data = np.genfromtxt(path, dtype=float, skip_header=1, delimiter=',')
    
    return data.astype(np.float32)

def pandas_load():
    
    path = '/home/kxtz813/test/test.csv'
    
    data = pd.read_csv(path, sep=',', header=None, engine='pyarrow', skiprows=1)
    
    return np.array(data).astype(np.float32)

def visualise_ft():
    
    image1 = '/home/kxtz813/test_im/half1.tif'
    image2 = '/home/kxtz813/test_im/half2.tif'
    
    im1, im2 = tiff.imread(image1), tiff.imread(image2)
    
    im1_fft, im2_fft = fft2(im1), fft2(im2)
    
    im1_fft, im2_fft = np.fft.fftshift(im1_fft), np.fft.fftshift(im2_fft)
    
    im1_fft, im2_fft = im1_fft.astype(np.complex64), im2_fft.astype(np.complex64)
    
    tiff.imsave('/home/kxtz813/test_im/ace2_half1_ft.tif',
              np.log(np.abs(im1_fft)))
    
    tiff.imsave('/home/kxtz813/test_im/ace2_half2_ft.tif',
               np.log(np.abs(im2_fft)))
    
    num = np.real(im1_fft * np.conj(im2_fft))

    denom1 = np.sqrt(np.abs(im1_fft**2))
    denom2 = np.sqrt(np.abs(im2_fft**2))
    
    tiff.imsave('/home/kxtz813/test_im/ace2_simple_split_numerator.tif',
               np.log(np.abs(num)))
    
    tiff.imsave('/home/kxtz813/test_im/ace2_denom_half1.tif',
               np.log(denom1))
    tiff.imsave('/home/kxtz813/test_im/ace2_denom_half2.tif',
               np.log(denom2))

if __name__ == "__main__":
    #import timeit
    #print(timeit.timeit("numpy_load()", number=100, setup="from __main__ import numpy_load"))
    visualise_ft()