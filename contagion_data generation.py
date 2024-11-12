import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import string
import itertools
import os
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


def lat_long_to_xyz(lat, lon):
    # Convert latitude and longitude to x, y, z coordinates on a sphere (Earth)
    R = 6371  # Earth's radius in kilometers
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return x, y, z


def assign_sensitive_attribute(num_v, vi_list, vc, m):
    # Compute D
    D = sum(num_v.values()) + len(num_v)
    # Calculate probabilities
    probabilities = []
    for vi in vi_list:
        random_number = random.uniform(1, round(m / 2))
        if vi == vc:
            numerator = (num_v[vi] + random_number)
        else:
            numerator = (num_v[vi] + 1)
        probabilities.append(numerator)
    probabilities = np.array(probabilities)
    S = probabilities.sum()
    probabilities /= S
    # Sample vi based on probabilities
    assigned_vi = np.random.choice(vi_list, p=probabilities)
    num_v[assigned_vi] += 1
    return assigned_vi


def trial01():
    n = 200000  # Number of points
    n_centers = 50  # Number of centers

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

        # Generate random centers within NYC
        center_latitudes, center_longitudes = generate_nyc_coordinates_by_real_density(n_centers)
        center_x, center_y, center_z = lat_long_to_xyz(center_latitudes, center_longitudes)
        centers = np.column_stack((center_x, center_y, center_z))

        # Assign each center a sensitive attribute value
        center_vi = np.random.choice(vi_list, n_centers)

        # Assign each point to the nearest center
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers)
        points = df[['X', 'Y', 'Z']].values
        distances, indices = nbrs.kneighbors(points)
        df['CenterIndex'] = indices.flatten()
        df['CenterVI'] = center_vi[indices.flatten()]

        # Assign sensitive attribute to each point
        num_v = {vi: 0 for vi in vi_list}
        sensitive_attributes = []
        for idx, row in df.iterrows():
            vc = row['CenterVI']
            assigned_vi = assign_sensitive_attribute(num_v, vi_list, vc, m)
            sensitive_attributes.append(assigned_vi)
            if idx % 10000 == 0 and idx > 0:
                print(f"Processed {idx} records")

        df['SensitiveAttribute'] = sensitive_attributes

        # Drop temporary columns
        df.drop(columns=['CenterIndex', 'CenterVI'], inplace=True)
        # Add a new column named 'root' with all values set to 'root'
        df['root'] = 'root'

        # Save to CSV
        filename = f'trial01/trial01_m_{m}.csv'

        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}\n")


def trial02():
    n_centers = 50  # Number of centers
    m = 15
    n = 200000  # Number of points
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

    # Generate random centers within NYC
    center_latitudes, center_longitudes = generate_nyc_coordinates_by_real_density(n_centers)
    center_x, center_y, center_z = lat_long_to_xyz(center_latitudes, center_longitudes)
    centers = np.column_stack((center_x, center_y, center_z))

    # Assign each center a sensitive attribute value
    center_vi = np.random.choice(vi_list, n)

    # Assign each point to the nearest center
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers)
    points = df[['X', 'Y', 'Z']].values
    distances, indices = nbrs.kneighbors(points)
    df['CenterIndex'] = indices.flatten()
    df['CenterVI'] = center_vi[indices.flatten()]

    # Assign sensitive attribute to each point
    num_v = {vi: 0 for vi in vi_list}
    sensitive_attributes = []
    for idx, row in df.iterrows():
        vc = row['CenterVI']
        assigned_vi = assign_sensitive_attribute(num_v, vi_list, vc, m)
        sensitive_attributes.append(assigned_vi)
        if idx % 10000 == 0 and idx > 0:
            print(f"Processed {idx} records")

    df['SensitiveAttribute'] = sensitive_attributes

    # Drop temporary columns
    df.drop(columns=['CenterIndex', 'CenterVI'], inplace=True)
    # Add a new column named 'root' with all values set to 'root'
    df['root'] = 'root'

    # Save to CSV
    filename = f'trial02/trial02.csv'

    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}\n")


def trial03():
    n_centers = 50  # Number of centers
    m = 15
    for size in range(100000, 500001, 100000):
        # Generate sensitive attribute values
        vi_list = generate_sensitive_values(m)

        # Generate IDs
        ids = np.arange(size)

        # Generate NYC coordinates based on detailed population density distribution
        latitudes, longitudes = generate_nyc_coordinates_by_real_density(n_centers)

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

        # Generate random centers within NYC
        center_latitudes, center_longitudes = generate_nyc_coordinates_by_real_density(n_centers)
        center_x, center_y, center_z = lat_long_to_xyz(center_latitudes, center_longitudes)
        centers = np.column_stack((center_x, center_y, center_z))

        # Assign each center a sensitive attribute value
        center_vi = np.random.choice(vi_list, n_centers)

        # Assign each point to the nearest center
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centers)
        points = df[['X', 'Y', 'Z']].values
        distances, indices = nbrs.kneighbors(points)
        df['CenterIndex'] = indices.flatten()
        df['CenterVI'] = center_vi[indices.flatten()]

        # Assign sensitive attribute to each point
        num_v = {vi: 0 for vi in vi_list}
        sensitive_attributes = []
        for idx, row in df.iterrows():
            vc = row['CenterVI']
            assigned_vi = assign_sensitive_attribute(num_v, vi_list, vc, m)
            sensitive_attributes.append(assigned_vi)
            if idx % 10000 == 0 and idx > 0:
                print(f"Processed {idx} records")

        df['SensitiveAttribute'] = sensitive_attributes

        # Drop temporary columns
        df.drop(columns=['CenterIndex', 'CenterVI'], inplace=True)
        # Add a new column named 'root' with all values set to 'root'
        df['root'] = 'root'

        # Save to CSV
        filename = f'trial02/trial03_size_{size}.csv'

        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}\n")


def main():
    trial01()  # Trial 1 -number of sensitive attribute values (200,000 data items, k=1000)
    trial02()  # Trial 2 -numbers of representatives k (200,000 data items, m=15)
    trial03()  # Trial 3 -different dataset sizes (k=1000, m=15)


if __name__ == "__main__":
    main()
