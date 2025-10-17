#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:33:22 2025

@author: kxtz813
"""

import numpy as np
import argparse
import os
import frc_comp_analysis.file_io as io
from frc_comp_analysis.frc_calculation import frc_fixed, frc_sigma
import frc_comp_analysis.plot_frc as frcplt


def check_args(args: object):
    """
    Check user-specified arguments. Checks:
    1) If arguments are missing.
    2) Existence of input folders and output folder
    3) Numerical validity of magnification
    4) Validity of splitting method.
    5) Validity of resolution criterion.
    """
    arg_dict = vars(args)

    for arg in arg_dict.values():
        if not arg:
            raise TypeError("One or more arguments missing.")

    if not os.path.isdir(arg_dict["data_folder"]):
        raise FileNotFoundError("Input file does not exist")

    if not os.path.isdir(arg_dict["output_folder"]):
        raise FileNotFoundError("Output folder does not exist")

    if arg_dict["magnification"] <= 0:
        raise ValueError("Magnification cannot be zero or negative")

    if arg_dict["split_method"] not in ("simple", "odd_even"):
        raise NameError(
            'Invalid data splitting method. Use either "simple" or "odd_even"'
        )

    if arg_dict["criterion"] not in ("fixed", "3sigma"):
        raise NameError('Invalid resolution criterion. Use only "fixed" or "3sigma"')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--magnification", type=float)
    parser.add_argument("--split_method", type=str)
    parser.add_argument("--criterion", type=str)

    opt = parser.parse_args()

    check_args(opt)

    data = io.load_reconstructions(opt.data_folder)

    print("The localisation files are: \n")
    print(data)

    frcs = np.zeros((len(data), 1))

    for i in range(0, len(data)):
        print("\n Processing dataset" + str(i + 1) + "\n")

        locs = io.load_localizations(data[i])

        if opt.criterion == "fixed":
            frc, spatial_freq, res, frc_res = frc_fixed(
                locs, opt.magnification, opt.split_method
            )

            frcplt.plot_frc_single(
                frc, spatial_freq, res, frc_res, i + 1, opt.output_folder
            )

        elif opt.criterion == "3sigma":
            frc, spatial_freq, res, frc_res, sig_curve = frc_sigma(
                locs, opt.magnification, opt.split_method
            )

            frcplt.plot_frc_single_sigma(
                frc, spatial_freq, res, frc_res, i + 1, opt.output_folder, sig_curve
            )

        frcs[i, 0] = res

        io.save_resolution(opt.output_folder, res, "noisy", index=i + 1)

        print("Processed dataset " + str(i + 1) + "\n")

    io.save_mean_frcs_single(frcs, opt.output_folder)

    all_data = io.save_frc_results_single(frcs, data, opt.output_folder)

    frcplt.plot_all_single(all_data, opt.output_folder)

    print("FRC calculations complete! The data are saved in -> " + opt.output_folder)


if __name__ == "__main__":
    main()
