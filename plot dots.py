import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = 'Data/Point.csv'
dtype = {'ID': 'int32', 'Latitude': 'float64', 'Longitude': 'float64'}
data = pd.read_csv(file_path, dtype=dtype)

# List of IDs to be highlighted in red
fileB_path = 'Data/Point_kmeans.csv'
df = pd.read_csv(fileB_path,encoding='ISO-8859-1')
A =set(df["cluster"])
 # replace with your actual IDs, using a set for faster lookup

# Separate the data into two DataFrames
highlighted_data = data[data['ID'].isin(A)]
other_data = data[~data['ID'].isin(A)]

# Create the plot
plt.figure(figsize=(10, 8))

# Plot non-highlighted points with hexbin to show density using a logarithmic color scale
hb = plt.hexbin(other_data['Longitude'], other_data['Latitude'], gridsize=50, cmap='Blues', bins='log', mincnt=1, alpha=0.6)

# Plot highlighted points on top
plt.scatter(highlighted_data['Longitude'], highlighted_data['Latitude'], c='red', s=10, edgecolors='red', alpha=1, label='Highlighted Points')

# Add labels and title
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Geographical Points Visualization')
#plt.legend()

# Show the plot
plt.show()
