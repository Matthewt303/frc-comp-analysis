#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:33:39 2025

@author: kxtz813
"""

import os
import pandas as pd
import numpy as np


def load_reconstructions(folder_path: str) -> list:
    folder_files = os.listdir(folder_path)

    reconstructions = [
        os.path.join(folder_path, file)
        for file in folder_files
        if file.endswith(".csv")
    ]

    return sorted(reconstructions)


def load_localizations(file: str) -> "np.ndarray":
    """
    This function opens the file containing localisation data and returns
    the txy-positions.
    -----------------------
    IN:
    file - the name of the localisation file
    ----------------------
    OUT:
    xy localisation data
    ----------------------
    """

    loc_data = pd.read_csv(
        file, sep=",", usecols=[1, 2, 3], header=None, engine="pyarrow", skiprows=1
    )

    return np.array(loc_data).astype(np.float32)


def save_resolution(
    output_folder: str, resolution: float, datatype: str, index: int
) -> None:
    fourier_space_res = np.float64(1 / resolution)

    if datatype == "noisy":
        with open(
            os.path.join(output_folder, "noisy_resolution_" + str(index) + ".txt"), "w"
        ) as f:
            f.write(
                "The real space resolution of noisy dataset "
                + str(index)
                + " is: "
                + str(resolution)
                + " nm \n"
            )

            f.write(
                "The corresponding fourier resolution of noisy dataset "
                + str(index)
                + " is: "
                + str(fourier_space_res)
                + " nm^-1 "
            )

    elif datatype == "denoised":
        with open(
            os.path.join(output_folder, "denoised_resolution_" + str(index) + ".txt"),
            "w",
        ) as f:
            f.write(
                "The real space resolution of denoised dataset "
                + str(index)
                + " is: "
                + str(resolution)
                + " nm \n"
            )

            f.write(
                "The corresponding fourier resolution if noisy dataset "
                + str(index)
                + " is: "
                + str(fourier_space_res)
                + " nm^-1"
            )


def save_mean_frcs(
    raw_frc: "np.ndarray", denoised_frc: "np.ndarray", output_folder: str
) -> None:
    raw_mean, denoised_mean = np.mean(raw_frc), np.mean(denoised_frc)
    raw_sd, denoised_sd = np.std(raw_frc, ddof=1), np.std(denoised_frc, ddof=1)

    with open(os.path.join(output_folder, "mean_frcs.txt"), "w") as f:
        f.write(
            "The mean FRC resolution for non-denoised data is: "
            + str(raw_mean)
            + " nm \n"
        )
        f.write(
            "The mean FRC resolution for denoised data is: "
            + str(denoised_mean)
            + " nm \n"
        )

        f.write(
            "The standard deviation for non-denoised data is: " + str(raw_sd) + " nm \n"
        )
        f.write(
            "The standard deviation for denoised data is: "
            + str(denoised_sd)
            + " nm \n"
        )

def save_mean_frcs_single(frc: "np.ndarray", output_folder: str) -> None:

    all_mean = np.mean(frc)
    all_sd = np.std(frc, ddof=1)

    with open(os.path.join(output_folder, "mean_frcs.txt"), "w") as f:
        f.write(
            "The mean FRC resolution for the data is: "
            + str(all_mean)
            + " nm \n"
        )

        f.write(
            "The standard deviation for the data is: " + str(all_sd) + " nm \n"
        )


def save_p_value(p_value: float, output_folder: str) -> None:
    with open(os.path.join(output_folder, "p_value.txt"), "w") as f:
        f.write("The p-value is: " + str(p_value))


def save_frc_results(
    raw_frc: "np.ndarray",
    denoised_frc: "np.ndarray",
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

    all_frc_results = np.vstack((raw_frc, denoised_frc))

    dataframe = pd.DataFrame(all_frc_results, columns=["FRC resolution (nm)"])

    dataframe.insert(0, "Designation", designations)
    dataframe.insert(0, "File name", all_files)

    dataframe.to_csv(os.path.join(output_folder, "all_results.csv"), index=False)

    return dataframe

def save_frc_results_single(frcs: "np.ndarray",
    loc_files: list,
    output_folder: str,
) -> "pd.DataFrame":
    
    dataframe = pd.DataFrame(frcs, columns=["FRC resolution (nm)"])

    dataframe.insert(0, "Blank", [""] * len(loc_files))
    dataframe.insert(0, "File name", loc_files)

    dataframe.to_csv(os.path.join(output_folder, "all_results.csv"), index=False)

    return dataframe