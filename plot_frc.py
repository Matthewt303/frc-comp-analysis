#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:13:07 2025

@author: kxtz813
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import seaborn as sns
import pandas as pd
import os


def plot_frc(frc_raw: 'np.ndarray', v_raw: 'np.ndarray',
             frc_denoised: 'np.ndarray', v_denoised: 'np.ndarray',
             output_folder: str, res_raw: float, res_raw_frc: float,
             res_denoised: float, res_denoised_frc: float,
             index: int, threshold:float=0.143) -> None:
    
    """Needs changing"""
    
    threshold_plot = np.full(frc_raw.shape[0], threshold)
    
    res_raw_v, res_dn_v = 1 / res_raw, 1 / res_denoised
    
    mpl.rcParams['font.sans-serif'] = ['Nimbus Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28
    
    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)
    
    ax.plot(v_raw, frc_raw, 'darkmagenta', label='Original', linewidth=4.5)
    ax.plot(v_denoised, frc_denoised, 'salmon', label='Denoised', linewidth=4.5)
    ax.plot(v_raw, threshold_plot, 'royalblue', label='Threshold', linewidth=4.5)
    ax.plot(res_raw_v, res_raw_frc, 'blueviolet', marker='.', markersize=24, markeredgecolor='k')
    ax.plot(res_dn_v, res_denoised_frc, 'blueviolet', marker='.', markersize=24, markeredgecolor='k')
    
    
    leg = plt.legend(loc='upper right')
    
    for line in leg.get_lines():
        line.set_linewidth(3.5)
    
    for text in leg.get_texts():
        text.set_fontsize(28)
    
    all_frc = np.vstack((frc_raw.reshape(-1, 1), frc_denoised.reshape(-1, 1)))
    
    ax.set_ylim(bottom=np.min(all_frc), top=1)
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
    
    plt.savefig(os.path.join(output_folder, 'frc_plot_' + str(index) + '.png'))
    plt.savefig(os.path.join(output_folder, 'frc_plot_' + str(index) + '.svg'))

def plot_all(all_data: 'pd.DataFrame', output_folder: str) -> None:
    
    """Boxplot or other"""
    
    plt.ioff()
    
    mpl.rcParams['font.sans-serif'] = ['Nimbus Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28
    
    sns.set_style('ticks')
    
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    
    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    graph = sns.stripplot(x=all_data.columns[1], y=all_data[all_data.columns[2]],
                          data=all_data, s=15, color='midnightblue')
    
    sns.pointplot(data=all_data, x=all_data.columns[1], y=all_data[all_data.columns[2]],
                  errorbar='sd', markers='_', linestyles='none', capsize=0.2,
                  linewidth=4.0, color='darkgreen')
    
    graph.tick_params(labelsize=30, pad=4)
    
    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

    ax.yaxis.set_minor_locator(AutoMinorLocator(11))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

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

    ax.set_xlabel('Data type', labelpad=6, fontsize=36)
    ax.set_ylabel(all_data.columns[2], labelpad=3, fontsize=36)
    
    plt.savefig(os.path.join(output_folder, 'all_frcs_plot.png'))
    plt.savefig(os.path.join(output_folder, 'all_frcs_plot.svg'))

def plot_frc_sigma(frc_raw: 'np.ndarray', v_raw: 'np.ndarray',
             frc_denoised: 'np.ndarray', v_denoised: 'np.ndarray',
             output_folder: str, res_raw: float, res_raw_frc: float,
             res_denoised: float, res_denoised_frc: float,
             sigma_curve_n: 'np.ndarray', sigma_curve_dn: 'np.ndarray',
             index: int) -> None:
    
    res_raw_v, res_dn_v = 1 / res_raw, 1 / res_denoised
    
    mpl.rcParams['font.sans-serif'] = ['Nimbus Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28
    
    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)
    
    ax.plot(v_raw, frc_raw, 'darkmagenta', label='Noisy', linewidth=4.5)
    ax.plot(v_denoised, frc_denoised, 'salmon', label='Denoised', linewidth=4.5)
    ax.plot(v_raw, sigma_curve_n, 'royalblue', label='3-σ curve noisy', linewidth=4.5)
    ax.plot(v_denoised, sigma_curve_dn, 'lightseagreen', label='3-σ curve denoised', linewidth=4.5)
    ax.plot(res_raw_v, res_raw_frc, 'blueviolet', marker='.', markersize=24, markeredgecolor='k')
    ax.plot(res_dn_v, res_denoised_frc, 'blueviolet', marker='.', markersize=24, markeredgecolor='k')
    
    
    leg = plt.legend(loc='upper right')
    
    for line in leg.get_lines():
        line.set_linewidth(3.5)
    
    for text in leg.get_texts():
        text.set_fontsize(28)
    
    all_frc = np.vstack((frc_raw.reshape(-1, 1), frc_denoised.reshape(-1, 1)))
    
    ax.set_ylim(bottom=np.min(all_frc), top=1)
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
    
    plt.savefig(os.path.join(output_folder, 'frc_plot_' + str(index) + '.png'))
    plt.savefig(os.path.join(output_folder, 'frc_plot_' + str(index) + '.svg'))

def plot_frc_single(frc: 'np.ndarray', v: 'np.ndarray',
                    res: float, res_frc: float, index: int,
                    output_folder: str, threshold: float=0.143) -> None:
    
    threshold_plot = np.full(frc.shape[0], threshold)
    
    res_v = 1 / res
    
    mpl.rcParams['font.sans-serif'] = ['Nimbus Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28
    
    fig, ax = plt.subplots(figsize=(11, 11), dpi=500)
    
    ax.plot(v, frc, 'darkmagenta', label='FRC', linewidth=4.5)
    ax.plot(v, threshold_plot, 'royalblue', label='Threshold', linewidth=4.5)
    ax.plot(res_v, res_frc, 'blueviolet', marker='.', markersize=24, markeredgecolor='k')
    
    
    leg = plt.legend(loc='upper right')
    
    for line in leg.get_lines():
        line.set_linewidth(3.5)
    
    for text in leg.get_texts():
        text.set_fontsize(28)
    
    ax.set_ylim(bottom=np.min(frc), top=1)
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
    
    plt.savefig(os.path.join(output_folder, 'frc_plot_' + str(index) + '.png'))
    plt.savefig(os.path.join(output_folder, 'frc_plot_' + str(index) + '.svg'))
    