#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:33:39 2025

@author: kxtz813
"""

import os
import pandas as pd
import numpy as np


def write_args(args: object, out: str) -> None:
    """
    Saves arguments as a .txt folder.
    -----------------------
    IN:
    args - user-specified input arguments
    out - the folder path where the .txt file will be saved
    ----------------------
    OUT:
    None
    ----------------------
    """
    arg_dict = vars(args)

    with open(os.path.join(out, "arguments.txt"), "w") as f:
        for arg, arg_val in zip(arg_dict.keys(), arg_dict.values()):
            f.write("The " + arg + " is " + str(arg_val) + "\n")


def load_reconstructions(folder_path: str) -> list[str]:
    """
    This function reads the folder where localization files are stored and converts
    it into a list of file names where each file is the absolute path of a
    localization table.
    -----------------------
    IN:
    folder_path - the name of the folder
    ----------------------
    OUT:
    reconstructions - list of absolute paths, sorted alphabetically.
    ----------------------
    """
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
    output_folder: str,
    resolution: float,
    datatype: str,
    index: int,
    condition_a: str,
    condition_b: str,
) -> None:
    """
    This function saves the FRC resolution in a .txt file. The text changes
    depending on whether the resolution is calculated from a 'noisy' (condition A)
    or a 'denoised' (condition B) dataset.
    -----------------------
    IN:
    output_folder - the name of the folder where the resolutions will be saved.
    resolution - the FRC resolution in nanometers.
    datatype - either 'cond_a' or 'cond_b'.
    index - the dataset number.
    condition_a - the designation for the control.
    condition_b - the designation for the test data.
    ----------------------
    OUT:
    None - but a file is saved.
    ----------------------
    """

    fourier_space_res = np.float64(1 / resolution)

    if datatype == "cond_a":
        with open(
            os.path.join(
                output_folder, condition_a + "_resolution_" + str(index) + ".txt"
            ),
            "w",
        ) as f:
            f.write(
                "The real space resolution of "
                + condition_a
                + " dataset "
                + str(index)
                + " is: "
                + str(resolution)
                + " nm \n"
            )

            f.write(
                "The corresponding fourier resolution of "
                + condition_a
                + " dataset "
                + str(index)
                + " is: "
                + str(fourier_space_res)
                + " nm^-1 "
            )

    elif datatype == "cond_b":
        with open(
            os.path.join(
                output_folder, condition_b + "_resolution_" + str(index) + ".txt"
            ),
            "w",
        ) as f:
            f.write(
                "The real space resolution of "
                + condition_b
                + " dataset "
                + str(index)
                + " is: "
                + str(resolution)
                + " nm \n"
            )

            f.write(
                "The corresponding fourier resolution of "
                + condition_b
                + " dataset "
                + str(index)
                + " is: "
                + str(fourier_space_res)
                + " nm^-1"
            )


def save_mean_frcs(
    raw_frc: "np.ndarray",
    denoised_frc: "np.ndarray",
    cond_a: str,
    cond_b: str,
    output_folder: str,
) -> None:
    """
    This function saves the mean FRC resolution in a .txt file for the noisy data
    and denoised data, as well as the standard deviation.
    -----------------------
    IN:
    raw_frc - the FRC resolutions for condition A
    denoised_frc - the FRC resolutions for condition B
    cond_a - the designation for the control.
    cond_b - the designation for the test data.
    output_folder - where the .txt files will be saved.
    ----------------------
    OUT:
    None - but a file is saved.
    ----------------------
    """
    raw_mean, denoised_mean = np.mean(raw_frc), np.mean(denoised_frc)
    raw_sd, denoised_sd = np.std(raw_frc, ddof=1), np.std(denoised_frc, ddof=1)

    with open(os.path.join(output_folder, "mean_frcs.txt"), "w") as f:
        f.write(
            "The mean FRC resolution for "
            + cond_a
            + " data is: "
            + str(raw_mean)
            + " nm \n"
        )
        f.write(
            "The mean FRC resolution for "
            + cond_b
            + " data is: "
            + str(denoised_mean)
            + " nm \n"
        )

        f.write(
            "The standard deviation for "
            + cond_a
            + " data is: "
            + str(raw_sd)
            + " nm \n"
        )
        f.write(
            "The standard deviation for "
            + cond_b
            + " data is: "
            + str(denoised_sd)
            + " nm \n"
        )


def save_mean_frcs_single(frc: "np.ndarray", output_folder: str) -> None:
    """
    This function saves the mean FRC resolution in a .txt file for one set of data
    as well as the standard deviation.
    -----------------------
    IN:
    frc - the FRC resolutions
    output_folder - where the .txt files will be saved.
    ----------------------
    OUT:
    None - but a file is saved.
    ----------------------
    """
    all_mean = np.mean(frc)
    all_sd = np.std(frc, ddof=1)

    with open(os.path.join(output_folder, "mean_frcs.txt"), "w") as f:
        f.write("The mean FRC resolution for the data is: " + str(all_mean) + " nm \n")

        f.write("The standard deviation for the data is: " + str(all_sd) + " nm \n")


def save_p_value(p_value: float, output_folder: str) -> None:
    """
    This function saves the p-value in a .txt file from a Mann-Whitney U-test
    between the noisy FRC resolutions and the denoised FRC resolutions.
    -----------------------
    IN:
    p_value - pvalue from the significance test
    output_folder - where the .txt file will be saved.
    ----------------------
    OUT:
    None - but a file is saved.
    ----------------------
    """
    with open(os.path.join(output_folder, "p_value.txt"), "w") as f:
        f.write("The p-value is: " + str(p_value))


def save_frc_results(
    raw_frc: "np.ndarray",
    denoised_frc: "np.ndarray",
    raw_loc_files: list[str],
    denoised_loc_files: list[str],
    cond_a: str,
    cond_b: str,
    output_folder: str,
) -> "pd.DataFrame":
    """
    This function converts the FRC results into a pandas dataframe where the columns
    are: file name, designation, and the FRC resolution. The dataframe is saved in the
    output folder as a .csv file.
    -----------------------
    IN:
    raw_frc - FRC resolutions from the noisy dataset
    denoised_frc - FRC resolutions from the denoised dataset.
    raw_loc_files - list of files for the noisy data
    denoised_loc_files - list of files for the denoised data.
    cond_a - the designation for the control.
    cond_b - the designation for the test data.
    output_folder - where the dataframe will be saved
    ----------------------
    OUT:
    dataframe - pandas dataframe with the files, condition, and FRC resolutions.
    ----------------------
    """
    designate_raw, designate_denoised = (
        [cond_a] * len(raw_loc_files),
        [cond_b] * len(denoised_loc_files),
    )

    designations = designate_raw + designate_denoised

    all_files = raw_loc_files.copy() + denoised_loc_files.copy()

    all_frc_results = np.vstack((raw_frc, denoised_frc))

    dataframe = pd.DataFrame(all_frc_results, columns=["FRC resolution (nm)"])

    dataframe.insert(0, "Designation", designations)
    dataframe.insert(0, "File name", all_files)

    dataframe.to_csv(os.path.join(output_folder, "all_results.csv"), index=False)

    return dataframe


def save_frc_results_single(
    frcs: "np.ndarray",
    loc_files: list[str],
    output_folder: str,
) -> "pd.DataFrame":
    """
    This function converts the FRC results into a pandas dataframe where the columns
    are: file name and the FRC resolution. The dataframe is saved in the
    output folder as a .csv file.
    -----------------------
    IN:
    frc - FRC resolutions from the noisy dataset
    loc_files - list of files for the data
    output_folder - where the dataframe will be saved
    ----------------------
    OUT:
    dataframe - pandas dataframe with the files, condition, and FRC resolutions.
    ----------------------
    """
    dataframe = pd.DataFrame(frcs, columns=["FRC resolution (nm)"])

    dataframe.insert(0, "Blank", [""] * len(loc_files))
    dataframe.insert(0, "File name", loc_files)

    dataframe.to_csv(os.path.join(output_folder, "all_results.csv"), index=False)

    return dataframe
