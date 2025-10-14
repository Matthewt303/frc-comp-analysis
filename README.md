## Overview

frc-comp-analysis is library for end-to-end Fourier ring correlation analysis
between two sets of single-molecule localization data or batch analysis for one set of localisation data. It is packaged as a python module that contains several command line scripts. 

## Prerequisites

- Python 3.11 or later
- A python virtual environment
- Two folders with at least two SMLM localisation files in each folder for
comparative analysis
- OR one folder with at least two SMLM localisation files for comparative analysis

## Installation

In an activated virtual environment:
```shell
pip install frc-comp-analysis
```

## Usage

### Command-line interface

It is recommended to run the analysis as a command line script. It can be run as:

```bash
comparative-analysis --data_folder --comp_folder --output_folder --magnification 
--split_method --criterion
```

Alternatively, in powershell:

```powershell
comparative-analysis.exe --data_folder --comp_folder --output_folder --magnification 
--split_method --criterion
```

### As a python script

```python
import frc-comp-analysis

```

## What to expect
For comparative analysis, the module should calculate the FRC resolution for each
localization file in both folders. It also generates publication-quality plots for
each pair of localization files as .png files and .svg files. Finally, it generates 
a dotplot with the FRC resolutions for both conditions along with summary statistics and the p-value from a Mann-Whitney U-test to assess for significant change in FRC resolutions between condition A and condition B.

Note, condition A is the 'control' and is designated 'noisy'. Condition B is designated
as 'denoised'. Originally, the module was intended to evaluate denoising but it can be used for any comparative analysis.