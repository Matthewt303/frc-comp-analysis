#!/home/kxtz813/.conda/envs/matthew_dev/bin/python3

import numpy as np
import argparse
import os
import FRC_analysis.file_io as io
from FRC_analysis.frc_calculation import frc_fixed, frc_sigma, calculate_p_value
from FRC_analysis.plot_frc import plot_frc, plot_all, plot_frc_sigma


def check_args(args: object) -> None:
    arg_dict = vars(args)

    if len(arg_dict) != 6:
        raise ValueError("Missing an input argument.")

    if not os.path.isdir(arg_dict["data_folder"]):
        raise FileNotFoundError("Input file does not exist")

    if not os.path.isdir(arg_dict["output_folder"]):
        raise FileNotFoundError("Output folder does not exist")

    if not os.path.isdir(arg_dict["denoised_folder"]):
        raise FileNotFoundError("Specified output folder does not exist.")

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
    parser.add_argument("--denoised_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--magnification", type=float)
    parser.add_argument("--split_method", type=str)
    parser.add_argument("--criterion", type=str)

    opt = parser.parse_args()

    check_args(opt)

    noisy_data = io.load_reconstructions(opt.data_folder)
    denoised_data = io.load_reconstructions(opt.denoised_folder)

    print("The noisy localisation files are: \n")
    print(noisy_data)

    print("The denoised localisation files are: \n")
    print(denoised_data)

    noisy_frcs = np.zeros((len(noisy_data), 1))
    denoised_frcs = np.zeros((len(denoised_data), 1))

    for i in range(len(noisy_data)):
        print("\n Processing dataset" + str(i + 1) + "\n")

        noisy_locs = io.load_localizations(noisy_data[i])
        denoised_locs = io.load_localizations(denoised_data[i])

        if opt.criterion == "fixed":
            frc_noisy, spatial_freq_noisy, res_noisy, frc_res_noisy = frc_fixed(
                noisy_locs, opt.magnification, opt.split_method
            )

            frc_denoised, spatial_freq_denoised, res_denoised, frc_res_denoised = (
                frc_fixed(denoised_locs, opt.magnification, opt.split_method)
            )

            plot_frc(
                frc_noisy,
                spatial_freq_noisy,
                frc_denoised,
                spatial_freq_denoised,
                opt.output_folder,
                res_noisy,
                frc_res_noisy,
                res_denoised,
                frc_res_denoised,
                index=i + 1,
            )

        elif opt.criterion == "3sigma":
            frc_noisy, spatial_freq_noisy, res_noisy, frc_res_noisy, sig_curve = (
                frc_sigma(noisy_locs, opt.magnification, opt.split_method)
            )

            (
                frc_denoised,
                spatial_freq_denoised,
                res_denoised,
                frc_res_denoised,
                sig_curve_dn,
            ) = frc_sigma(denoised_locs, opt.magnification, opt.split_method)

            plot_frc_sigma(
                frc_noisy,
                spatial_freq_noisy,
                frc_denoised,
                spatial_freq_denoised,
                opt.output_folder,
                res_noisy,
                frc_res_noisy,
                res_denoised,
                frc_res_denoised,
                sig_curve,
                sig_curve_dn,
                index=i + 1,
            )

        noisy_frcs[i, 0] = res_noisy
        denoised_frcs[i, 0] = res_denoised

        io.save_resolution(opt.output_folder, res_noisy, "n", index=i + 1)
        io.save_resolution(opt.output_folder, res_denoised, "dn", index=i + 1)

        print("Processed dataset " + str(i + 1) + "\n")

    io.save_mean_frcs(noisy_frcs, denoised_frcs, opt.output_folder)

    all_data = io.save_frc_results(
        noisy_frcs, denoised_frcs, noisy_data, denoised_data, opt.output_folder
    )

    p_value_test = calculate_p_value(noisy_frcs, denoised_frcs)
    io.save_p_value(p_value_test, opt.output_folder)

    plot_all(all_data, opt.output_folder)

    print("FRC calculations complete! The data are saved in -> " + opt.output_folder)


if __name__ == "__main__":
    main()
