import pandas as pd
from anytree import Node, RenderTree

# Load the CSV file
file_path = "Data/Temp.csv"
df = pd.read_csv(file_path)

# Create the root node
root = Node("root")

# Create dictionaries to hold nodes for each level
continents = {}
countries = {}
cities = {}

# Create a dictionary to store counts
counts = df.groupby(['root', 'Continent', 'Country', 'City']).size().reset_index(name='Count')

# Populate the tree with nodes and add occurrences
for _, row in counts.iterrows():
    root_name = row['root']
    continent_name = row['Continent']
    country_name = row['Country']
    city_name = row['City']
    count = row['Count']

    # Add continent node if not exists
    if continent_name not in continents:
        continent_count = (counts[counts['Continent'] == continent_name]['Count'].sum()/117415)*500
        continents[continent_name] = Node(f"{continent_name} ({continent_count})", parent=root)

    # Add country node if not exists
    if country_name not in countries:
        country_count = (counts[counts['Country'] == country_name]['Count'].sum()/117415)*500
        countries[country_name] = Node(f"{country_name} ({country_count})", parent=continents[continent_name])

    # Add city node
    Node(f"{city_name} ({(count/117415)*500})", parent=countries[country_name])

# Render the tree
for pre, fill, node in RenderTree(root):
    print(f"{pre}{node.name}")
