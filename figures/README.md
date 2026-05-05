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

## Basic Model Structure and Requirements:

The model runs in python and especially requires the package Gillespy2. Input variables to the model include A1 (initial effective ligand concentration), AF (final effective ligand concentration), cell number, cell type, numbers of receptors, and number of timepoints in the simulation. For each figure, the stable variables are generally hardcoded, while the updating variables are set as system arguments, to be used in the simulation as well as the naming structure of the output data file.

## Figure 1

Figure made in Inkscape (open-source vector graphics editor). Schematic structure based on Karim 2021 under CC BY 4.

## Figure 2

a. created using the deterministic model from Karim 2021. concentrations: [.001, .002, .003, .004, .005, .006, .007, .008, .009, .01, .02, .03, .04, .05, .06, .07, .08, .09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0] File location: "graph2A_matlab.m" <br>
b. stochastic model run with same concentrations as 2a. receptor levels set at 3500 for each type I and 7000 for type II. File location: "fig2b_graph_matla.m" <br>
c. coefficient of variation claculated for each concentration used in 2b. Standard deviation/average tetramer complex level. File location: "fig2c_graph_matlab.m" <br>
d. gradient shape approximated based on general BMP gradient as experimentally suggested through pSmad measurement about leading edge of zebrafish embryo. File location: "fig2f_matlab.m" <br>
e. 36 cell simulations were run for each of the 5 shown concentrations. The listed concentration was used for the cell with maximum concentration and the next cells followed the BMP gradient shape from 2d. <br>
f.

## Figure 3

a. <br>
b. <br>
c. <br>
d. <br>

## Figure 4

a-c. <br>

## Figure 5

a-c.
d-i.
