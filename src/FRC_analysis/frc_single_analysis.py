#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:33:22 2025

@author: kxtz813
"""

import numpy as np
import argparse
import os
import file_io as io
from frc_calculation import frc_fixed
from plot_frc import plot_frc_single, plot_all


def check_args(args: object):
    arg_dict = vars(args)

    if len(arg_dict) != 4:
        raise ValueError("Missing an input argument.")

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

            plot_frc_single(frc, spatial_freq, res, frc_res, i + 1, opt.output_folder)

        frcs[i, 0] = res

        io.save_resolution(opt.output_folder, res, "noisy", index=i + 1)

        print("Processed dataset " + str(i + 1) + "\n")

    io.save_mean_frcs(frcs, frcs.copy(), opt.output_folder)

    all_data = io.save_frc_results(
        frcs, frcs.copy(), data, data.copy(), opt.output_folder
    )

    plot_all(all_data, opt.output_folder)

    print("FRC calculations complete! The data are saved in -> " + opt.output_folder)


if __name__ == "__main__":
    main()
