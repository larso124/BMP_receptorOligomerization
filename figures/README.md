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
f. EC50 concentration (concentration at which half max activation occurs) was found from 2b data. This concentration was used for each receptor as the maximum value in a BMP gradient. 36 cells were simulated with decrasing concentrations, following the BMP gradient shape from 2d. Each simulation was run for 240 hours. The graphs show average and shaded standard deviation.

## Figure 3

a. The 10 days of simulations from 1f were broken into 10 sample days. Fast Fourier Transform was used on each day-long tetramer level data set. Resulting frequencies were shown in a cummulative manner.<br>
b. Following the same process as 3a, the resulting frequencies were binned to show relative contribution.<br>
c. A random 6-hours of data were pulled for each receptor from their max concentration cell from the 10-days of simulation. Using the averages and standard deviations from the full 10-day simulation of the cell at maximum concentration, the 6-hours of data were centered about the mean and departures greater than 1 standard deviation were highlighted.<br>
d. The full 10-day simulations for the top concentration group of cells were parsed to find any continuous departures greated than one standard deivation from the mean and track their lengths. The lengths were binned and shown in bar graph form.<br>

## Figure 4

a-c. 36-cell 24-hour simulations were run for each receptor relative EC50 concentration for each receptor component fold change. Only one value was changed relative to the original simulation - that was to each individual receptor component level. Each resulting gradient profile was processed for positional information. <br>

## Figure 5

a-c. The three initial graphs are representative schematics of the regulation modes. <br>
d-i. Three hours of simulation were done to represent each line. The system was simulatied using the EC50 of the relative receptor as the final gradient maximum. Variables for this set of simulations included initial ligand concentration (A1), final ligand concentration (AF), and number of time steps (tps). Regulation mode was encoded in the initial and final concentrations but also passed as a string for file naming.
