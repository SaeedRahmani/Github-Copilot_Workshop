import numpy as np
import pandas as pd

def generate_data(n_samples=100):
    X = np.random.rand(n_samples, 1)
    noise = np.random.randn(n_samples, 1)
    Y = 4 * X + 2 + noise
    data = np.hstack((X, Y))
    df = pd.DataFrame(data, columns=['X', 'Y'])
    df.to_csv('results/manual_data.csv', index=False)

generate_data()

def generate_data(a, b, n_samples=100):
    X = np.random.rand(n_samples, 1)
    noise = np.random.randn(n_samples, 1)
    Y = a * X + b + noise
    data = np.hstack((X, Y))
    df = pd.DataFrame(data, columns=['X', 'Y'])
    df.to_csv('results/user_data.csv', index=False)

generate_data(4, 2)