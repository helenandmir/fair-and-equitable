import pandas as pd
from geopy.geocoders import Nominatim
import time

# Load the full CSV file
file_path = 'Data/Fully_Cleaned_Updated_DCA_Legally_Operating_Businesses.csv'
df = pd.read_csv(file_path)

# Create a sample of 100 rows from the dataframe
sample_df = df.sample(n=1000, random_state=1)

# Initialize the geocoder
geolocator = Nominatim(user_agent="geoapiExercises")

# Function to get latitude and longitude
def get_lat_long(row):
    try:
        address = f"{row['STREET']}, {row['CITY']}, {row['ZIP']}"
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        return None, None

# Apply the function to the sample dataframe
sample_df['latitude'], sample_df['longitude'] = zip(*sample_df.apply(get_lat_long, axis=1))

# Save the final geocoded dataframe to a new CSV file
output_file_path = 'Data/Geocoded_Sample_1000_Fully_Cleaned_DCA_Legally_Operating_Businesses.csv'
sample_df.to_csv(output_file_path, index=False)

print(f"Geocoding completed. File saved to {output_file_path}")
