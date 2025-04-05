import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load heatmap data from files
path_coverage = np.loadtxt('path_coverage_matrix1b.txt')
trash_collection = np.loadtxt('trash_collection_matrix1b.txt')

# Display settings from main.py
maxPercentage = 60
paddingMultiplyer = 1

# Get data dimensions and calculate aspect ratio
rows, cols = path_coverage.shape
aspect_ratio = cols / rows

# Display final averaged heatmaps in 1x3 horizontal arrangement
base_height = 8  # Reduced height to allow more relative spacing
fig_width = base_height * aspect_ratio * 3  # Width for 3 columns
fig_height = base_height  # Single row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))

# Set larger font sizes for better readability
plt.rcParams.update({
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7
})
barsize = 0.08

# Path coverage heatmap (averaged)
im1 = ax1.imshow(path_coverage, cmap='hot', interpolation='nearest',
                norm=plt.Normalize(vmin=0, vmax=np.max(path_coverage)*paddingMultiplyer))
ax1.set_title("Average Path Coverage")
fig.colorbar(im1, ax=ax1, label="Average Coverage", fraction=barsize, aspect=20)

# Trash collection heatmap (averaged)
im2 = ax2.imshow(trash_collection, cmap='hot', interpolation='nearest', 
                norm=plt.Normalize(vmin=0, vmax=maxPercentage*paddingMultiplyer))
ax2.set_title("Average Trash Collected")
fig.colorbar(im2, ax=ax2, label="Average Percentage", fraction=barsize, aspect=20)

# Correlation analysis and heatmap
# Flatten and filter non-zero values
actual = trash_collection[trash_collection > 0].flatten()
predicted = path_coverage[trash_collection > 0].flatten() * (maxPercentage/np.max(path_coverage))

# Calculate Pearson correlation and p-value
r, p_value = stats.pearsonr(actual, predicted)
print(f"Correlation between actual and predicted values:")
print(f"Pearson r: {r:.3f}, p-value: {p_value:.3e}")

# Create correlation heatmap
corr_heatmap = (path_coverage/np.max(path_coverage)) * (trash_collection/np.max(trash_collection))
im3 = ax3.imshow(corr_heatmap, cmap='hot', interpolation='nearest',
                vmin=0, vmax=1)  # Scale from 0 to 1 since we're multiplying two normalized maps
ax3.set_title(f'Correlation Heatmap')
plt.colorbar(im3, ax=ax3, label='Product (0-1 scale)', fraction=barsize, aspect=20)

plt.tight_layout(pad=4.0, w_pad=7.0, h_pad=3.0)  # More spacing between subplots
plt.show()
