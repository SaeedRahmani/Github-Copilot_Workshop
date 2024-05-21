import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# import statements

def fit_linear_regression():
    """
    Fits a linear regression model to the data in the 'results' directory.

    Returns:
    - coefficients (list): List of regression coefficients for each file.
    - residuals (list): List of mean squared errors for each file.
    """
    files = os.listdir('results')  # Get the list of files in the 'results' directory
    csv_files = [f for f in files if f.endswith('.csv')]  # Filter the list to include only CSV files

    coefficients = []  # Initialize an empty list to store the regression coefficients
    residuals = []  # Initialize an empty list to store the mean squared errors

    for file in csv_files:
        df = pd.read_csv(os.path.join('results', file))  # Read the CSV file into a pandas DataFrame
        X = df.iloc[:, 0].values.reshape(-1, 1)  # Extract the X values from the DataFrame
        Y = df.iloc[:, 1].values.reshape(-1, 1)  # Extract the Y values from the DataFrame

        model = LinearRegression()  # Create a LinearRegression model
        model.fit(X, Y)  # Fit the model to the data

        Y_pred = model.predict(X)  # Predict the Y values using the fitted model
        residual = mean_squared_error(Y, Y_pred)  # Calculate the mean squared error

        coefficients.append(model.coef_[0][0])  # Append the regression coefficient to the coefficients list
        residuals.append(residual)  # Append the mean squared error to the residuals list

    return coefficients, residuals  # Return the coefficients and residuals lists

coef, res = fit_linear_regression()  # Call the fit_linear_regression function and store the results in coef and res variables

print("Coefficients:", coef)  # Print the coefficients list
print("Residuals:", res)  # Print the residuals list

import matplotlib.pyplot as plt

def plot_data_and_model():
    """
    Plots the data and fitted model for each file in the 'results' directory.
    Saves the plots as PNG files in the 'results' directory.
    """
    coefficients, residuals = fit_linear_regression()  # Get the coefficients and residuals using the fit_linear_regression function
    files = os.listdir('results')  # Get the list of files in the 'results' directory
    csv_files = [f for f in files if f.endswith('.csv')]  # Filter the list to include only CSV files

    for i, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join('results', file))  # Read the CSV file into a pandas DataFrame
        X = df.iloc[:, 0].values.reshape(-1, 1)  # Extract the X values from the DataFrame
        Y = df.iloc[:, 1].values.reshape(-1, 1)  # Extract the Y values from the DataFrame

        plt.figure()  # Create a new figure
        plt.scatter(X, Y, color='blue')  # Plot the data points
        plt.plot(X, coefficients[i]*X, color='red')  # Plot the fitted model
        plt.title('Data and Fitted Model')  # Set the title of the plot
        plt.xlabel('X')  # Set the label for the X-axis
        plt.ylabel('Y')  # Set the label for the Y-axis
        plt.savefig(f'results/plot_{i}.png')  # Save the plot as a PNG file
        plt.show()  # Show the plot

plot_data_and_model()  # Call the plot_data_and_model function
