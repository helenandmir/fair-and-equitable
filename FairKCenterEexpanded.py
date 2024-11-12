import math
import time
import csv
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
from collections import defaultdict
from scipy.spatial import cKDTree
from collections import Counter
import statistics


class FairKCenter:
    def __init__(self, req_dic, input_fileA, input_fileB, attributs_list, k, f):
        self.fileA = input_fileA  # Information about the points, including coordinates and sensitive attributes (color)
        self.fileB = input_fileB  # The neighborhood radius of each point
        self.req_dic = req_dic
        self.req_dic_save = req_dic
        self.f = f  #
        self.k = k  # Number of reps
        csv_file = self.filter_colors(self.fileA, list(self.req_dic.keys()))
        NR_temp = self.get_f_feasible_neighborhood_radius(csv_file, self.f)
        self.df = pd.read_csv(input_fileB, encoding='ISO-8859-1')
        self.attributs_list = attributs_list

        # Dictionaries where the keys are  ids of the points:
        self.NR_dic = dict(sorted(NR_temp.items(), key=lambda item: item[1]))  # The values are the neighborhood radius
        self.coordinates_dic = self.get_coordinates_from_csv(self.fileB)  # The values are the coordinates (x,y,z)
        self.list_of_attributs_dic = self.get_colors_from_csv(self.fileB)  # The values are the colors
        self.distance_to_reps_dic = {}  # The values are the distance of each point to the reps set
        self.closest_rep_dic = {}
        self.alpha_dic = {}
        self.Latitude = dict(self.df["Latitude"])
        self.Longitude = dict(self.df["Longitude"])

        # Dictionaries where the keys are the representatives:
        self.rep_to_color_dic = {}  # The values are the sensitive attribute (color) of each rep
        self.balls_dic = {}  # The values are a list of all the points that each rep ball contains
        self.balls_colors_dic = {}  # The values are a list of all the colors that each rep ball contains

        self.point_list = list(self.NR_dic.keys())  # All the points ids

    # Filters columns in a CSV file based on provided colors, saving the result to a new file.
    def filter_colors(self, input_file, colors):
        # Read the original CSV file
        df = pd.read_csv(input_file)

        # Selecting columns of colors that are in the provided colors list
        selected_columns = ['ID', 'NR_TYPE_ONE'] + [color for color in colors if color in df.columns]

        # Creating a new DataFrame with only the selected columns
        filtered_df = df[selected_columns]

        # Define the name of the output file based on the input file name
        file_name, file_extension = os.path.splitext(input_file)
        output_file = file_name + '_new' + file_extension

        # Save the DataFrame to a new CSV file
        filtered_df.to_csv(output_file, index=False)

        return output_file

    # Calculates the f-feasible neighborhood radius for each point in the given CSV file,
    # and returns a dictionary containing the point IDs as keys and their
    # f-feasible neighborhood radius as values
    def get_f_feasible_neighborhood_radius(self, csv_file, f):
        # Dictionary to store the closest distances for each point
        closest_distances = defaultdict(list)

        # Reading the CSV file and loading the data
        with open(csv_file, newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skipping the header row
            for row in reader:
                if (row[0] == ''):
                    break
                point_id, *distances = row
                dis = [float(d) for d in distances]
                distances = dis[1:]
                closest = sorted(distances)[:f]
                closest.append(dis[0])
                closest = sorted(closest)[:f + 1]
                for d in closest:
                    closest_distances[point_id].append(d)

        # Dictionary to store the farthest distances among the closest distances
        farthest_distances = {}

        for point_id, distances in closest_distances.items():
            farthest_distances[int(float(point_id))] = max(distances)

        return farthest_distances

    # Extracts coordinates (x, y, z) from the given CSV file, assuming the columns are labeled 'ID', 'X', 'Y', and 'Z'.
    # Returns a dictionary where the keys are point IDs and the values are tuples containing the coordinates.
    def get_coordinates_from_csv(self, csv_file):
        points_dict = {}
        with open(csv_file) as file:
            csv_reader = csv.reader(file)
            # Get the header row
            header = next(csv_reader)

            # Find the indices of the 'id', 'x', 'y', and 'z' columns
            id_index = header.index('ID')
            x_index = header.index('X')
            y_index = header.index('Y')
            z_index = header.index('Z')

            for row in csv_reader:
                point_id = row[id_index]
                x = float(row[x_index])
                y = float(row[y_index])
                z = float(row[z_index])
                points_dict[int(point_id)] = (x, y, z)
        return points_dict

    # Extracts colors from the given CSV file, assuming the columns are labeled 'ID' and 'Colors'.
    # Returns a dictionary where the keys are point IDs and the values are the corresponding colors.
    def get_colors_from_csv(self, csv_file):
        list_of_dict = []
        for i in self.attributs_list:
            temp_dic = {}
            with open(csv_file, 'r') as file:
                csv_reader = csv.reader(file)
                # Get the header row
                header = next(csv_reader)
                att = header.index(i)

                # Find the indices of the 'id' and 'color' columns
                id_index = header.index('ID')
                for row in csv_reader:
                    point_id = row[id_index]
                    temp_dic[int(point_id)] = row[att]
            list_of_dict.append(temp_dic)

        return list_of_dict

    def get_ball_colors(self, ball_list, p):
        # Creating an empty set to store the unique colors
        unique_colors_set = set()
        p_x, p_y, p_z = self.coordinates_dic[p]
        # For each point in the list of points
        for point in ball_list:
            x, y, z = self.coordinates_dic[point]
            distance = math.sqrt((x - p_x) ** 2 + (y - p_y) ** 2 + (z - p_z) ** 2)
            self.distance_to_reps_dic[point] = distance
            for att_d in self.list_of_attributs_dic:
                # Assigning the color of the point to the variable color
                color = att_d[point]
                # Adding the color to the set of unique colors
                if color in self.req_dic:
                    unique_colors_set.add(color)

        # Returning the list of unique colors
        return list(unique_colors_set)

    # Checks whether a given point 'point_p' can be considered a new representative
    # among a list of points ('point_list') based on their coordinates and neighborhood radii.
    # Returns True if the distance between 'point_p' and any point in 'point_list'
    # is greater than the sum of their respective neighborhood radii, otherwise returns False.
    def is_p_new_rep(self, point_list, point_p, alpha):
        p_x, p_y, p_z = self.coordinates_dic[point_p]
        for point in point_list:
            x, y, z = self.coordinates_dic[point]
            distance = math.sqrt((x - p_x) ** 2 + (y - p_y) ** 2 + (z - p_z) ** 2)
            if distance <= self.NR_dic[point_p] + self.NR_dic[point]:  # alpha*
                self.distance_to_reps_dic[point_p] = distance
                return False

        return True

    def two_fair(self, alpha):
        tree = KDTree(np.array(list(self.coordinates_dic.values())))
        temp_NR_dic = self.NR_dic
        centers_list = []
        while len(temp_NR_dic) != 0:

            p = next(iter(temp_NR_dic.keys()))

            if self.is_p_new_rep(centers_list, p, alpha):
                centers_list.append(p)
                delete_points = tree.query_ball_point(self.coordinates_dic[p], temp_NR_dic[p])  # + 0.0001
                delete_points_set = set(delete_points)
                temp_NR_dic = {key: temp_NR_dic[key] for key in temp_NR_dic if key not in delete_points_set}
                self.balls_dic[p] = delete_points_set
                self.balls_colors_dic[p] = self.get_ball_colors(delete_points_set, p)

                if len(self.balls_colors_dic[p]) == 0:
                    print("p = {}, ball = {}".format(p, self.balls_colors_dic[p]))
            else:

                del temp_NR_dic[p]

        return centers_list

    def updatr_dict(self, dic_new_reps_colors, req_dic):
        B = {}
        for value in dic_new_reps_colors.values():
            B[value] = B.get(value, 0) + 1

        for v in req_dic.keys():
            if v in B.keys():
                req_dic[v] = req_dic[v] - B[v]

        return req_dic

    def update_distances_with_kdtree(self, p_id):
        # Get the coordinates of point p
        p_coord = np.array(self.coordinates_dic[p_id])

        # Prepare a list of all point IDs and their coordinates, including p
        all_point_ids = list(self.coordinates_dic.keys())
        all_coords = np.array([self.coordinates_dic[pid] for pid in all_point_ids])

        # Compute distances using vectorized operations
        deltas = all_coords - p_coord  # Shape: (N, D)
        distances = np.linalg.norm(deltas, axis=1)  # Shape: (N,)

        # Update the dictionaries
        for pid, dist in zip(all_point_ids, distances):
            current_dist = self.distance_to_reps_dic.get(pid, float('inf'))
            if dist < current_dist:
                self.distance_to_reps_dic[pid] = dist
                self.closest_rep_dic[pid] = p_id

    def build_unique_values_dict(self):
        # Read the header of the CSV to get available columns
        df = pd.read_csv(self.fileB)

        # Filter A to include only columns present in the CSV

        columns_to_read = ["Colors", "root"]
        # columns_to_read = [col for col in self.attributs_list if col in csv_columns]

        if not columns_to_read:
            return {}

        # Initialize a dictionary to store unique values
        unique_values_dict = {col: set(df[col]) for col in columns_to_read}
        unique_values_dict2 = {}
        for att in unique_values_dict:
            unique_values_dict2[att] = []
            for v in unique_values_dict[att]:
                if (v in self.req_dic.keys() and self.req_dic[v] > 0):
                    unique_values_dict2[att].append(v)

        return unique_values_dict2

    # Completion for k representatives
    def select_points_as_new_agents(self, colors_dict, points_dict, distances_dict):
        selected_points = []
        dic_att_to_values = self.build_unique_values_dict()
        for att in dic_att_to_values.values():
            for val in att:
                if val in points_dict:
                    for i in range(1, colors_dict[val] + 1):
                        p_id = max(points_dict[val], key=lambda k: self.distance_to_reps_dic[k])
                        self.rep_to_color_dic[p_id] = val
                        selected_points.append(p_id)
                        self.update_distances_with_kdtree(p_id)

        if len(self.rep_to_color_dic.keys()) != self.k:
            print("why??")

        return selected_points

    def closest_agent_distances(self, D, A, B):
        # Creating a list of coordinates for the agent points from the list B
        agents_coords = [D[id_] for id_ in B]

        # Creating a KD tree from the agent points in list B
        tree = cKDTree(agents_coords)

        # Finding the closest agent for each point in list A
        for id_ in A:
            # If the point doesn't exist in dictionary D
            if id_ not in D:
                print(f"Point with ID {id_} not found in the dictionary D.")
                continue
            if id_ in B:
                self.closest_rep_dic[id_] = id_
                self.distance_to_reps_dic[id_] = 0
                continue
            # Finding the closest agent to the point from list A
            closest_agent_id = tree.query([D[id_]], k=1)[1][0]

            # Calculating the distance between the point and the closest agent
            closest_distance = tree.query([D[id_]], k=1)[0][0]

            # Adding the closest agent to the dictionary for the current ID
            self.closest_rep_dic[id_] = B[closest_agent_id]
            # Adding the distance to the dictionary for the current ID
            self.distance_to_reps_dic[id_] = closest_distance

    def get_alpha_dic(self, A, B):
        # Iterate over the keys in dictionary A
        for id_, value_A in A.items():
            # Check if the key exists in dictionary B
            if id_ in B:
                # Divide the value from A by the value from B and store the result in the result dictionary
                self.alpha_dic[id_] = value_A / B[id_]

    def calculate_points_stddev_on_sphere(self):
        # Create a counter to count the occurrences of each agent
        agent_counter = Counter(self.closest_rep_dic.values())
        print(agent_counter)
        num_point_in_ball = list(agent_counter.values())
        std = statistics.stdev(num_point_in_ball)
        max_load = max(agent_counter.values())

        return std, max_load

    def print_result(self):

        # Finding the maximum value in the dictionary
        max_dis = max(self.distance_to_reps_dic.values())

        # Finding the key corresponding to the maximum value
        max_p = [key for key, value in self.distance_to_reps_dic.items() if value == max_dis][0]

        print("The greatest distance between a point to its rep is: d(p={},rep={})={} ".format(max_p,
                                                                                               self.closest_rep_dic[
                                                                                                   max_p], max_dis))

        # Finding the maximum value in the dictionary
        max_dis = max(self.alpha_dic.values())

        # Finding the key corresponding to the maximum value
        max_p = [key for key, value in self.alpha_dic.items() if value == max_dis][0]

        print("The biggest alpha is: d(p={},rep={})/NR(p={}) ={}".format(max_p, self.closest_rep_dic[max_p], max_p,
                                                                         max_dis))

        # Finding the maximum NR
        max_dis = max(self.NR_dic.values())
        # Finding the key corresponding to the maximum value
        max_p = [key for key, value in self.NR_dic.items() if value == max_dis][0]
        print("The biggest NR is: NR({}) ={}".format(max_p, max_dis))
        stv, max_load = self.calculate_points_stddev_on_sphere()
        avg = len(self.point_list) / self.k
        print("The standard deviation of the number of points belonging to each rep is: {}".format(stv))
        print("The largest number of points that belong to that representative is: {} ".format(max_load))
        print("The optimal number of points belonging to each rep is: {}".format(avg))

    def invert_dictionary(self, A):
        # Initialize an empty dictionary to store the inverted results
        inverted_dict = {}

        # Iterate over each point ID and representative ID in the dictionary A
        for point_id, representative_id in A.items():
            # If the representative ID is not already in the inverted dictionary, add it with an empty list
            if representative_id not in inverted_dict:
                inverted_dict[representative_id] = []
            # Append the current point ID to the list of the representative ID in the inverted dictionary
            inverted_dict[representative_id].append(point_id)

        return inverted_dict

    def update_reps_with_kdtree(self, new_reps):
        """
        Updates the distance_to_reps_dic and closest_rep_dic dictionaries so that the value of each point
        in distance_to_reps_dic is the distance to the closest point in new_reps, and the value in
        closest_rep_dic is the ID of the closest point in new_reps.

        Parameters:
        - new_reps: List of point IDs that are the new representatives.
        - distance_to_reps_dic: Dictionary mapping point IDs to their distance from their representative.
        - closest_rep_dic: Dictionary mapping point IDs to the ID of their closest representative.
        - coordinates_dic: Dictionary mapping point IDs to their coordinates.
        """
        # Extract coordinates of new representatives
        rep_coords = np.array([self.coordinates_dic[rep_id] for rep_id in new_reps])
        rep_ids = np.array(new_reps)

        # Build KDTree with the coordinates of new representatives
        tree = cKDTree(rep_coords)

        # Extract coordinates of all points
        all_point_ids = list(self.coordinates_dic.keys())
        all_point_coords = np.array([self.coordinates_dic[pid] for pid in all_point_ids])

        # Query the tree to find the closest representative for each point
        distances, indices = tree.query(all_point_coords)

        # Update the dictionaries
        for pid, dist, idx in zip(all_point_ids, distances, indices):
            closest_rep_id = rep_ids[idx]
            self.distance_to_reps_dic[pid] = dist
            self.closest_rep_dic[pid] = closest_rep_id

    def count_unique_values_in_groups(self, groups):
        # Read the CSV file

        # Create a set of unique values from the columns city, county, continent
        unique_values = set([])
        for att in self.attributs_list:
            unique_values = unique_values.union(set(self.df[att]))

        # Create a DataFrame for the result
        result_df = pd.DataFrame(columns=['group'] + list(unique_values))
        # Initialize row index
        row_index = 0

        # Count the values in each group and add them to the DataFrame
        for group, indices in groups.items():
            group_data = self.df.iloc[indices]
            value_counts = Counter()
            for att in self.attributs_list:
                value_counts.update(group_data[att])

            row = {'group': group}
            for value in unique_values:
                row[value] = value_counts[value]

            # Add the row to the DataFrame using loc
            result_df.loc[row_index] = row
            row_index += 1

        # Write the result to a new CSV file
        result_df.to_csv('group_POI.csv', index=False)
