import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the Excel file
df = pd.read_excel('steel_fibers/steel_fibers.xlsx')

# Create KDE for each feature
kde_diameter = stats.gaussian_kde(df['average_diameter'])
kde_length = stats.gaussian_kde(df['length'])

# Generate random samples from the distributions
n_samples = 10000
random_diameter = kde_diameter.resample(n_samples)[0]
random_length = kde_length.resample(n_samples)[0]

# Plot the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original data histograms
axes[0, 0].hist(df['average_diameter'], bins='auto', alpha=0.7, density=True, color='skyblue')
axes[0, 0].set_title('Original Average Diameter')
axes[0, 1].hist(df['length'], bins='auto', alpha=0.7, density=True, color='lightcoral')
axes[0, 1].set_title('Original Length')

# KDE fits and random samples
x_diam = np.linspace(df['average_diameter'].min(), df['average_diameter'].max(), 1000)
x_len = np.linspace(df['length'].min(), df['length'].max(), 1000)

axes[1, 0].plot(x_diam, kde_diameter(x_diam), 'b-', lw=2, label='KDE')
axes[1, 0].hist(random_diameter, bins='auto', alpha=0.5, density=True, color='lightblue', label='Random Samples')
axes[1, 0].set_title('KDE Fit for Average Diameter')
axes[1, 0].legend()

axes[1, 1].plot(x_len, kde_length(x_len), 'r-', lw=2, label='KDE')
axes[1, 1].hist(random_length, bins='auto', alpha=0.5, density=True, color='lightpink', label='Random Samples')
axes[1, 1].set_title('KDE Fit for Length')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Generated {n_samples} random samples for each feature")
print(f"Average diameter samples range: {random_diameter.min():.2f} to {random_diameter.max():.2f}")
print(f"Length samples range: {random_length.min():.2f} to {random_length.max():.2f}")