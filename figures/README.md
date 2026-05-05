# Figure Guide:

| Figure No. | Subfigure | Description                                                   | Program  |
| ---------- | --------- | ------------------------------------------------------------- | -------- |
| 1          |           | Model Interactions and Species                                | Inkscape |
| 2          | a         | Deterministic Model                                           | MATLAB   |
|            | b         | Stochastic Model                                              | MATLAB   |
|            | c         | Coefficient of Variation                                      | MATLAB   |
|            | d         | Gradient Shape                                                | MATLAB   |
|            | e         | Positional Information from Profiles                          | MATLAB   |
|            | f         | Steady-State Tetramer Gradients                               | MATLAB   |
| 3          | a         | Cumulative FFT                                                | MATLAB   |
|            | b         | Binned FFT Contribution                                       | MATLAB   |
|            | c         | Shaded Error                                                  | MATLAB   |
|            | d         | Departure/Error Lengths                                       | MATLAB   |
| 4          | a-c       | Positional Information with Individual Receptor Levels Scaled | MATLAB   |
| 5          | a-c       | Gradient Development Schematics                               | Inkscape |
|            | d-i       | Positional Information under Dynamic Gradients                | MATLAB   |

# Basic Model Structure and Requirements:

The model runs in python and especially requires the package Gillespy2. Input variables to the model include A1 (initial effective ligand concentration), AF (final effective ligand concentration), cell number, cell type, numbers of receptors, and number of timepoints in the simulation. For each figure, the stable variables are generally hardcoded, while the updating variables are set as system arguments, to be used in the simulation as well as the naming structure of the output data file.

# Figure 1
