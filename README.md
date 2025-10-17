## Overview

frc-comp-analysis is a library for end-to-end Fourier ring correlation analysis
between two sets of single-molecule localization data or batch analysis for one set of localisation data. It is packaged as a python module that contains several command line scripts. 

The FRC calculation is based on [1](https://www.nature.com/articles/nmeth.2448):

## Prerequisites

- Python 3.11 or later
- A python virtual environment
- Two folders with at least two SMLM localisation files in each folder for
comparative analysis
- OR one folder with at least two SMLM localisation files for comparative analysis
- SMLM localisation file(s) as .csv, with frames, x, and y in columns 2, 3, and 4, respectively.

## Installation

For use as a command-line script, create a virtual environment then run:
```shell
git clone https://github.com/Matthewt303/frc-comp-analysis.git
cd frc-comp-analysis
python3 -m pip install .
```
For Windows Powershell, use ```python``` instead of ```python3```.

Following this ```comparative-analysis``` and ```batch-analysis``` should become available as scripts.

## Usage

### Command-line

It is recommended to run the analysis as a command line script. It can be run as:

```bash
comparative-analysis --data_folder /path/to/condition_A --comp_folder /path/to/condition_B --output_folder /path/to/output --magnification 0.1 
--split_method odd_even --criterion fixed
```

Alternatively, in powershell:

```powershell
comparative-analysis.exe --data_folder \path\to\condition_A --comp_folder \path\to\condition_B --output_folder \path\to\output --magnification 0.1 
--split_method odd_even --criterion fixed
```

For batch analysis:

```bash
batch-analysis --data_folder /path/to/data --output_folder /path/to/output --magnification 0.1 --split_method odd_even --criterion fixed
```

## Parameters

A brief summary of parameters is given here:

- --magnification: scaling factor for super-resolution image. Between 0.05 to 0.20 is recommended
- --split_method: how the dataset is split. Use 'simple' or 'odd_even'.
- --criterion: threshold for determining resolution. Use 'fixed' or '3sigma'.

For more detailed discussions on these parameters, please refer to the original paper and also [2](https://www.nature.com/articles/s41377-023-01321-0).

## What to expect
For comparative analysis, the module should calculate the FRC resolution for each
localization file in both folders. It also generates publication-quality plots for
each pair of localization files as .png files and .svg files. Finally, it generates 
a dotplot with the FRC resolutions for both conditions along with summary statistics and the p-value from a Mann-Whitney U-test to assess for significant change in FRC resolutions between condition A and condition B.

Note, condition A is the 'control' and is designated 'noisy'. Condition B is designated
as 'denoised'. Originally, the module was intended to evaluate denoising but it can be used for any comparative analysis.

## References

(1) Nieuwenhuizen, R. P. J., Lidke, K. A., Bates, M., Puig, D. L., Grünwald, D.,
Stallinga, S. & Rieger, B. Measuring image resolution in optical nanoscopy. *Nature
Methods* **10**, 557–562 (2013).

(2) Zhao, W., Huang, X., Yang, J., Qu, L., Qiu, G., Zhao, Y., Wang, X., Su, D.,
Ding, X., Mao, H., Jiu, Y., Hu, Y., Tan, J., Zhao, S., Pan, L., Chen, L. & Li,
H. Quantitatively mapping local quality of super-resolution microscopy by rolling
Fourier ring correlation. *Light: Science & Applications* **12**, 298 (2023)
