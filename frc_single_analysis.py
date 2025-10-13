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
from frc_calculation import frc_workflow, calculate_p_value
from plot_frc import plot_frc_single

def check_args(args: object):
    
    arg_dict = vars(args)
    
    if len(arg_dict) != 4:
        
        raise ValueError('Missing an input argument.')
    
    if not os.path.isdir(arg_dict['data_folder']):
        
        raise FileNotFoundError('Input file does not exist')
        
    if not os.path.isdir(arg_dict['output_folder']):
        
        raise FileNotFoundError('Output folder does not exist')
    
    if arg_dict['magnification'] <= 0:
        
        raise ValueError('Magnification cannot be zero or negative')
    
    if arg_dict['frame_data'] not in ('y', 'n'):
        
        raise NameError('Invalid specification for inclusion of frame data. Use only (y/n)')

def main():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--magnification', type=float)
    parser.add_argument('--frame_data', type=str)
    
    opt = parser.parse_args()
    
    check_args(opt)
    
    data = io.load_reconstructions(opt.data_folder)
    
    print("The localisation files are: \n")
    print(data)
    
    frcs = np.zeros((len(data), 1))
    
    for i in range(0, len(data)):
        
        print("\n Processing dataset " + str(i + 1) + "\n")
        
        locs = io.load_localizations(data[i])
        
        frc, spatial_freq, res, frc_res = frc_workflow(locs, opt.magnification, opt.frame_data)
        
        frcs[i, 0] = res
        
        io.save_resolution(opt.output_folder, res, 'n', index=i + 1)
        
        plot_frc_single(frc, spatial_freq, res, frc_res, i + 1, opt.output_folder)
        
    io.save_mean_frcs(frcs, frcs, opt.output_folder)

if __name__ == "__main__":
    
    main()
