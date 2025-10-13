#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 09:01:05 2025

@author: kxtz813
"""

import numpy as np
import pandas as pd
import tifffile as tiff
import os
import argparse
from frc_calculation import calculate_p_value
from file_io import save_p_value
from plot_frc import plot_all

## Image loading ##


def collate_im_files(folder: str) -> list:
    folder_files = os.listdir(folder)

    im_files = [
        os.path.join(folder, file) for file in folder_files if file.endswith(".tif")
    ]

    return sorted(im_files)


def load_image(file_name: str) -> "np.ndarray":
    image = tiff.imread(file_name)

    return image.astype(np.float32)


## Image processing ##


def filt_image(image: "np.ndarray") -> "np.ndarray":
    pixels = image.copy().flatten()

    nonzero_pix = np.where(pixels > 0)

    return pixels[nonzero_pix]


def calculate_median_frc(image: "np.ndarray") -> float:
    return np.median(image)


def calculate_mean_frc(image: "np.ndarray") -> float:
    return np.mean(image)


def calculate_frc_std(image: "np.ndarray") -> float:
    return np.std(image)


def process_image(file_name: str) -> float:
    im = load_image(file_name)

    filt_im = filt_image(im)

    mean_rfrc = calculate_mean_frc(filt_im)
    median_rfrc = calculate_median_frc(filt_im)
    rfrc_std = calculate_frc_std(filt_im)

    return mean_rfrc, median_rfrc, rfrc_std


## File output ##


def save_results(
    raw_frcs: "np.ndarray",
    denoised_frcs: "np.ndarray",
    raw_loc_files: list,
    denoised_loc_files: list,
    output_folder: str,
) -> "pd.DataFrame":
    designate_raw, designate_denoised = (
        ["Noisy"] * len(raw_loc_files),
        ["Denoised"] * len(denoised_loc_files),
    )

    designations = designate_raw + designate_denoised

    all_files = raw_loc_files.copy() + denoised_loc_files.copy()

    all_frc_results = np.vstack((raw_frcs, denoised_frcs))

    print(all_frc_results.shape)

    columns = [
        "Mean rFRC resolution (nm)",
        "Median rFRC resolution (nm)",
        "Standard deviation (nm)",
    ]

    dataframe = pd.DataFrame(all_frc_results, columns=columns)

    dataframe.insert(0, "Designation", designations)
    dataframe.insert(0, "File name", all_files)

    dataframe.to_csv(os.path.join(output_folder, "all_results.csv"), index=False)

    return dataframe


def check_args(args: object) -> None:
    arg_dict = vars(args)

    if len(arg_dict) != 3:
        raise ValueError("Missing an input argument.")

    if not os.path.isdir(arg_dict["data_folder"]):
        raise FileNotFoundError("Input file does not exist")

    if not os.path.isdir(arg_dict["output_folder"]):
        raise FileNotFoundError("Output folder does not exist")

    if not os.path.isdir(arg_dict["denoised_folder"]):
        raise FileNotFoundError("Specified output folder does not exist.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--denoised_folder", type=str)
    parser.add_argument("--output_folder", type=str)

    opt = parser.parse_args()

    noisy_im_files = collate_im_files(opt.data_folder)
    denoised_im_files = collate_im_files(opt.denoised_folder)

    noisy_rfrcs = np.zeros((len(noisy_im_files), 3))
    denoised_rfrcs = np.zeros((len(denoised_im_files), 3))

    for i, file in enumerate(zip(noisy_im_files, denoised_im_files)):
        noisy_rfrcs[i, 0], noisy_rfrcs[i, 1], noisy_rfrcs[i, 2] = process_image(file[0])
        denoised_rfrcs[i, 0], denoised_rfrcs[i, 1], denoised_rfrcs[i, 2] = (
            process_image(file[1])
        )

    all_data = save_results(
        noisy_rfrcs,
        denoised_rfrcs,
        noisy_im_files,
        denoised_im_files,
        opt.output_folder,
    )

    pval = calculate_p_value(noisy_rfrcs[:, 0], denoised_rfrcs[:, 0])
    save_p_value(pval, opt.output_folder)

    plot_all(all_data, opt.output_folder)


if __name__ == "__main__":
    main()
