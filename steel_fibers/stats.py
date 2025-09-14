import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Load your Excel file
df = pd.read_excel("steel_fibers/steel_fibers.xlsx")  # replace with your filename
print(df.describe())
bw_multiplier = 0.2
bw_method = "silverman"
n_samples = 1000

# ---- Enhanced Random sample generator ----
def generate_samples_kde(column, n_samples=n_samples, bw_method=bw_method, bw_multiplier=bw_multiplier, min_value=0.05):
    # Ensure input is a 1D numpy array
    data = np.asarray(column).ravel()
    
    # Fit KDE with scipy.stats.gaussian_kde
    kde = gaussian_kde(data, bw_method=bw_method)
    
    # Apply bandwidth multiplier if specified
    if bw_multiplier != 1.0:
        # Use type ignore to suppress PyLance warning
        current_bandwidth = kde.factor * bw_multiplier  # type: ignore
        kde.set_bandwidth(current_bandwidth)
    
    # Sample from KDE until we have enough valid samples
    valid_samples = []
    samples_needed = n_samples
    
    while samples_needed > 0:
        # Generate samples in batches
        batch_samples = kde.resample(samples_needed * 2).ravel()  # Generate extra to account for filtering
        
        # Filter samples that meet the minimum value requirement
        valid_batch = batch_samples[batch_samples > min_value]
        
        # Add valid samples to our collection
        valid_samples.extend(valid_batch)
        samples_needed = n_samples - len(valid_samples)
    
    # Take exactly n_samples (in case we got more than needed)
    samples = np.array(valid_samples[:n_samples])
    
    return samples, kde

# Generate samples with adjustable parameters
random_diameters, kde_diam = generate_samples_kde(
    df['average_diameter'], 
    n_samples=n_samples, 
    bw_method=bw_method,  # Try "silverman" for different smoothing
    bw_multiplier=bw_multiplier   # Increase for smoother, decrease for more detailed
)

random_lengths, kde_length = generate_samples_kde(
    df['length'], 
    n_samples=n_samples,
    bw_method=bw_method,
    bw_multiplier=bw_multiplier
)

# Create wider evaluation ranges that cover both original and generated data
diam_min = min(df['average_diameter'].min(), random_diameters.min())
diam_max = max(df['average_diameter'].max(), random_diameters.max())
diam_range = np.linspace(diam_min, diam_max, 1000)

length_min = min(df['length'].min(), random_lengths.min())
length_max = max(df['length'].max(), random_lengths.max())
length_range = np.linspace(length_min, length_max, 1000)

# Plot original + generated histograms with KDE lines
plt.figure(figsize=(14, 5))

# --- Original average diameter ---
plt.subplot(2,2,1)
plt.hist(df['average_diameter'], bins=100, density=True, alpha=0.7, color='blue', label='Data')
plt.plot(diam_range, kde_diam(diam_range), 'r-', linewidth=2, label='KDE')
plt.title("Original Average Diameter")
plt.xlabel("Average Diameter")
plt.ylabel("Samples")
plt.legend()

# --- Generated average diameter ---
plt.subplot(2,2,2)
plt.hist(random_diameters, bins=100, density=True, alpha=0.7, color='orange', label='Generated')
plt.plot(diam_range, kde_diam(diam_range), 'r-', linewidth=2, label='KDE')  # Use the same KDE that generated the data
plt.title("Average Diameter - Generated")
plt.xlabel("Average Diameter")
plt.ylabel("Samples")
plt.legend()

# --- Original length ---
plt.subplot(2,2,3)
plt.hist(df['length'], bins=100, density=True, alpha=0.7, color='green', label='Data')
plt.plot(length_range, kde_length(length_range), 'r-', linewidth=2, label='KDE')
plt.title("Original Length")
plt.xlabel("Length")
plt.ylabel("Samples")
plt.legend()

# --- Generated length ---
plt.subplot(2,2,4)
plt.hist(random_lengths, bins=100, density=True, alpha=0.7, color='red', label='Generated')
plt.plot(length_range, kde_length(length_range), 'r-', linewidth=2, label='KDE')  # Use the same KDE that generated the data
plt.title("Length - Generated")
plt.xlabel("Length")
plt.ylabel("Samples")
plt.legend()

plt.tight_layout()
plt.show()

# Print bandwidth information for tuning
print(f"Diameter KDE bandwidth: {kde_diam.factor:.4f}")
print(f"Length KDE bandwidth: {kde_length.factor:.4f}")