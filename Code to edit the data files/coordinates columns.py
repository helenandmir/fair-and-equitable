import pandas as pd
import numpy as np

# Load the CSV file
file_path = 'Data/Updated_Businesses_with_Coordinates.csv'
df = pd.read_csv(file_path)

# Define the function to convert latitude and longitude to Cartesian coordinates
def lat_lon_to_cartesian(lat, lon):
    R = 6371  # Radius of the Earth in kilometers
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return x, y, z

# Apply the function to create X, Y, Z columns
df['X'], df['Y'], df['Z'] = zip(*df.apply(lambda row: lat_lon_to_cartesian(row['Latitude'], row['Longitude']), axis=1))

# Save the updated dataframe to a new CSV file
output_file_path = 'Data/Updated_Businesses_with_Coordinates2.csv'
df.to_csv(output_file_path, index=False)

print(f"Updated CSV file saved to {output_file_path}")
