import numpy as np
import pandas as pd
import string
import itertools
import random


def generate_sensitive_values(m):
    # Generate unique labels for sensitive attribute
    letters = list(string.ascii_uppercase)
    if m <= 26:
        return letters[:m]
    else:
        # Generate combinations like 'A', 'B', ..., 'Z', 'AA', 'AB', ...
        combinations = [''.join(comb) for i in range(1, 3) for comb in itertools.product(letters, repeat=i)]
        return combinations[:m]


def generate_nyc_coordinates_by_real_density(n):
    # Define a detailed population density distribution for NYC using smaller grid cells (synthetic data for illustration)
    nyc_grid_data = pd.DataFrame({
        'GridCell': [
            'Manhattan_Upper', 'Manhattan_Mid', 'Brooklyn_North', 'Brooklyn_South', 'Queens_North', 'Queens_South',
            'Bronx_West', 'Bronx_East', 'StatenIsland_North', 'StatenIsland_South'
        ],
        'Latitude': [
            40.7900, 40.7500, 40.7100, 40.6400, 40.7600, 40.7000,
            40.8500, 40.8200, 40.6300, 40.5500
        ],
        'Longitude': [
            -73.9660, -73.9900, -73.9400, -73.9500, -73.8000, -73.8200,
            -73.8800, -73.8500, -74.1000, -74.1500
        ],
        'PopulationDensity': [
            90000, 100000, 85000, 80000, 70000, 60000,
            75000, 72000, 20000, 15000
        ]
    })

    # Sample grid cells based on population density weights
    grid_probabilities = nyc_grid_data['PopulationDensity'] / nyc_grid_data['PopulationDensity'].sum()
    sampled_grid_cells = nyc_grid_data.sample(n, weights=grid_probabilities, replace=True)

    # Generate random offsets around each grid cell to simulate individual locations within NYC
    latitudes = sampled_grid_cells['Latitude'] + np.random.normal(0, 0.001, n)
    longitudes = sampled_grid_cells['Longitude'] + np.random.normal(0, 0.001, n)

    return latitudes.values, longitudes.values


def trial1():
    n = 200000  # Number of points

    for m in range(5, 51, 5):
        # Generate sensitive attribute values
        vi_list = generate_sensitive_values(m)

        # Generate IDs
        ids = np.arange(n)

        # Generate NYC coordinates based on detailed population density distribution
        latitudes, longitudes = generate_nyc_coordinates_by_real_density(n)

        # Convert to X, Y, Z coordinates
        x_coords, y_coords, z_coords = lat_long_to_xyz(latitudes, longitudes)

        # Create DataFrame
        df = pd.DataFrame({
            'ID': ids,
            'Latitude': latitudes,
            'Longitude': longitudes,
            'X': x_coords,
            'Y': y_coords,
            'Z': z_coords
        })

        # Assign sensitive attribute to each point uniformly
        df['SensitiveAttribute'] = np.random.choice(vi_list, n)

        # Add a new column named 'root' with all values set to 'root'
        df['root'] = 'root'

        # Save to CSV
        filename = f'trial1/trial1_m_{m}.csv'
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}\n")


def trial2():
    n = 200000  # Number of points
    m = 15
    # Generate sensitive attribute values
    vi_list = generate_sensitive_values(m)

    # Generate IDs
    ids = np.arange(n)

    # Generate NYC coordinates based on detailed population density distribution
    latitudes, longitudes = generate_nyc_coordinates_by_real_density(n)

    # Convert to X, Y, Z coordinates
    x_coords, y_coords, z_coords = lat_long_to_xyz(latitudes, longitudes)

    # Create DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'Latitude': latitudes,
        'Longitude': longitudes,
        'X': x_coords,
        'Y': y_coords,
        'Z': z_coords
    })

    # Assign sensitive attribute to each point uniformly
    df['SensitiveAttribute'] = np.random.choice(vi_list, n)

    # Add a new column named 'root' with all values set to 'root'
    df['root'] = 'root'

    # Save to CSV
    filename = f'trial2/trial2.csv'
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}\n")


def trial3():
    m = 15
    for n in range(100000, 500001, 100000):
        # Generate sensitive attribute values
        vi_list = generate_sensitive_values(m)

        # Generate IDs
        ids = np.arange(n)

        # Generate NYC coordinates based on detailed population density distribution
        latitudes, longitudes = generate_nyc_coordinates_by_real_density(n)

        # Convert to X, Y, Z coordinates
        x_coords, y_coords, z_coords = lat_long_to_xyz(latitudes, longitudes)

        # Create DataFrame
        df = pd.DataFrame({
            'ID': ids,
            'Latitude': latitudes,
            'Longitude': longitudes,
            'X': x_coords,
            'Y': y_coords,
            'Z': z_coords
        })

        # Assign sensitive attribute to each point uniformly
        df['SensitiveAttribute'] = np.random.choice(vi_list, n)

        # Add a new column named 'root' with all values set to 'root'
        df['root'] = 'root'

        # Save to CSV
        filename = f'trial3/trial3_size_{n}.csv'
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}\n")


def main():
    trial1()  # Trial 1 -number of sensitive attribute values (200,000 data items, k=1000)
    trial2()  # Trial 2 -numbers of representatives k (200,000 data items, m=15)
    trial3()  # Trial 3 -different dataset sizes (k=1000, m=15)


def lat_long_to_xyz(lat, lon):
    # Convert latitude and longitude to x, y, z coordinates on a sphere (Earth)
    R = 6371  # Earth's radius in kilometers
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return x, y, z


if __name__ == "__main__":
    main()
