# Linear Regression Project

This project generates data based on a linear function, fits a linear regression model on it, and plots both the data and the fitted model.

## Project Structure

- `README.md`: This file.
- `results/`: Contains the generated data and plots.
- `scripts/`: Contains the Python scripts for data generation and model fitting.

## Scripts

- `data_generation.py`: Generates data based on the linear function Y = 4X + 2 (plus some random noise) and saves it as CSV files in the `results` directory. The number of samples can be adjusted by changing the `n_samples` parameter when calling the `generate_data` function. [Link to file](scripts/data_generation.py)

- `fit.py`: Contains two functions:
    - `fit_linear_regression`: Reads the CSV files in the `results` directory, fits a linear regression model for each file, and returns the coefficients and residuals of each model. [Link to function](scripts/fit.py#L5)
    - `plot_data_and_model`: Reads all CSV files in the `results` directory, plots the data and the fitted model for each file, saves the plots in the `results` directory, and displays the plots. The red line in each plot represents the fitted model. [Link to function](scripts/fit.py#L20)

## Usage

1. Run `data_generation.py` to generate the data.
2. Run `fit.py` to fit the linear regression model and plot the data and the fitted model.