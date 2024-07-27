from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import time
import math
import pandas as pd
from bigtree import Node
import csv
import MaxMatching
from scipy.spatial import KDTree
from FairKCenterEexpanded import FairKCenter


# Function to invert the dictionary
def invert_dictionary(dictionary):
    inverted_dict = {}
    for key, value in dictionary.items():
        if value not in inverted_dict:
            inverted_dict[value] = [key]
        else:
            inverted_dict[value].append(key)
    return inverted_dict


    # Extracts colors from the given CSV file, assuming the columns are labeled 'ID' and 'Colors'.
    # Returns a dictionary where the keys are point IDs and the values are the corresponding colors.
def get_colors_from_csv(csv_file,attributs_list):
    list_of_dict =[]
    for i in attributs_list:
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

def get_ball_colors(ball_list,list_of_attributs_dic,req_dic):
        # Creating an empty set to store the unique colors
        unique_colors_set = set()

        # For each point in the list of points
        for point in ball_list:
            for att_d in list_of_attributs_dic:
                # If the point exists in the colors dictionary
                 if point in att_d:
                     # Assigning the color of the point to the variable color
                     color = att_d[point]
                     # Adding the color to the set of unique colors
                     if color in req_dic:
                         unique_colors_set.add(color)


        # Returning the list of unique colors
        return list(unique_colors_set)

def create_tree(df):
    nodes = {}
    children_count = {}

    for _, row in df.iterrows():
        levels = row.values

        for i in range(len(levels)):
            node_value = levels[i]
            parent_value = levels[i - 1] if i > 0 else None

            # Create node if it doesn't exist
            if node_value not in nodes:
                nodes[node_value] = Node(node_value, parent=nodes.get(parent_value, None))
                children_count[node_value] = 0

            # Update children count of the parent
            if parent_value:
                children_count[parent_value] += 1

    # Assign children and siblings count
    for node in nodes.values():
        if node.parent:
            node.children_count = children_count[node.name]
            node.siblings_count = len(node.parent.children)
        else:
            node.children_count = children_count[node.name]
            node.siblings_count = 0

    return nodes


def create_hierarchy_uniform(k,csv_file_path, columns):
    # Read data from CSV file
    df = pd.read_csv(csv_file_path)
    num_of_points = len(list(df["root"]))

    # Dictionary to store created nodes
    nodes_temp = create_tree(df[columns])
    nodes = {}



    # Create a new node for "root
    nodes["root"] = Node("root", eq=k, abs_eq=k, parent=None)

    for i in columns[1:]:
        list_of_val = list(df[i].unique())

        for val in list_of_val:
            parent_column = columns[columns.index(i) - 1]
            parent_value = df[df[i] == val][parent_column].values[0]
            parent_eq = nodes[parent_value].eq
            val_siblings_num = nodes_temp[val].siblings_count
            uni_eq = math.floor(parent_eq/val_siblings_num)
            num_of_point_val = len([j for j in df[i] if j == val])
            val_eq = min(uni_eq, num_of_point_val)
            nodes[val] = Node(val, eq=val_eq, abs_eq=val_eq,parent=nodes[parent_value])

    return nodes

def create_hierarchy_relative(k,csv_file_path, columns):
    # Read data from CSV file
    df = pd.read_csv(csv_file_path)
    num_of_points = len(list(df["root"]))

    # Dictionary to store created nodes
    nodes = {}

    # Iterate through each column in the hierarchical order
    for column in columns:
        for value in df[column].unique():
            parent_node = None

            num_of_value = len([i for i in df[column] if i==value])
            eq_num = math.floor((num_of_value/num_of_points)*k)

            # Check if there is a parent in the previous columns
            if columns.index(column) > 0:
                parent_column = columns[columns.index(column) - 1]
                parent_value = df[df[column] == value][parent_column].values[0]
                parent_node = nodes[parent_value]

            # Create a new node
            nodes[value] = Node(value,eq=eq_num,abs_eq =eq_num, parent=parent_node)

    return nodes


def update_abs_eq(list_of_constraints):
    dic_of_constraints ={}
    dic_of_constraints2 ={}

    for c in list_of_constraints.values():
        if c.name != "root":
           p = c.parent
           p.abs_eq = p.abs_eq - c.eq

    dic_of_constraints[list_of_constraints["root"].name] = list_of_constraints["root"].abs_eq
    dic_of_constraints2[list_of_constraints["root"].name] = list_of_constraints["root"].eq

    for c in list_of_constraints.values():
        dic_of_constraints[c.name] = c.abs_eq
        dic_of_constraints2[c.name] = c.eq
    return dic_of_constraints, dic_of_constraints2

def updatr_dict(list_reps,req_dic,list_of_attributs_dic):
        for att in list_of_attributs_dic:
            for rep in list_reps:

                if att[rep] in req_dic.keys() and req_dic[att[rep]] >=1:
                    req_dic[att[rep]]=req_dic[att[rep]]-1

        return {k: v for k, v in req_dic.items() if v != 0}



def closest_agent_distances(D, A, B):
        closest_rep_dic={}
        distance_to_reps_dic={}
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
                closest_rep_dic[id_] = id_
                distance_to_reps_dic[id_] = 0
                continue
            # Finding the closest agent to the point from list A
            closest_agent_id = tree.query([D[id_]], k=1)[1][0]

            # Calculating the distance between the point and the closest agent
            closest_distance = tree.query([D[id_]], k=1)[0][0]

            # Adding the closest agent to the dictionary for the current ID
            closest_rep_dic[id_] = B[closest_agent_id]
            # Adding the distance to the dictionary for the current ID
            distance_to_reps_dic[id_] = closest_distance
        with open('output.txt', 'a') as f:
            print("->>>closest_agent_distances<<<-", file=f)
        return closest_rep_dic,distance_to_reps_dic
def get_coordinates_from_csv(csv_file):
        points_dict = {}
        with open(csv_file, 'r') as file:
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

if __name__ == '__main__':
    start = time.time()
    attributs_list=["root","Colors"]#attributs_list
    fileA = 'Data/Point_kmeans.csv'
    fileB = 'Data/Point.csv'
    fileC = 'Data/Point_radius_500.csv'
    df1 = pd.read_csv(fileA)
    df2 = pd.read_csv(fileB)

    nodes = create_hierarchy_relative(500, fileB, ["root", "Colors"])
    dic_of_absEq, dic_of_Eq = update_abs_eq(nodes)
    print(dic_of_absEq)

    print(sum([i for i in dic_of_absEq.values()]))
    dic_of_constraints2 = {}#req_dic
    for i in dic_of_absEq.keys():
        if dic_of_absEq[i] > 0:
            dic_of_constraints2[i] = dic_of_absEq[i]

    dic_id_clusters = dict(zip(df1["ID"], df1["cluster"]))
    dic_centers_cluster = invert_dictionary(dic_id_clusters)#obj.balls_dic

    obj = FairKCenter(dic_of_constraints2, fileC, fileB, attributs_list, 500, 0)
    for rep_id in list(dic_centers_cluster.keys()):
        obj.balls_colors_dic[rep_id] = obj.get_ball_colors(dic_centers_cluster[rep_id])
        obj.balls_dic[rep_id] = dic_centers_cluster[rep_id]
    C = MaxMatching.CR(obj.req_dic, obj.balls_colors_dic, obj.list_of_attributs_dic, obj.balls_dic)
    new_reps, dic_new_reps_colors, num_colors_dic = C.create_graph()

    for i in dic_centers_cluster.keys():
        if i not in list(list(dict(new_reps).keys())+list(dict(new_reps).values())):
            new_reps.append((i,i))
    dict_new_rep=dict(new_reps)
    for point_id in list(df1["ID"]):
        obj.is_p_new_rep(list(dic_new_reps_colors.keys()), point_id, 1)

    obj.rep_to_color_dic.update(dic_new_reps_colors)

    obj.req_dic = obj.updatr_dict(dict_new_rep, obj.req_dic)
    dict_of_list_map_colors = obj.color_points_mapping(obj.req_dic)


    # Adding the new column with the mapped values
    df1['new_cluster'] = df1['cluster'].map(dict_new_rep)

    # Saving the updated DataFrame back to a new CSV file
    df1.to_csv(fileA, index=False)
    all_points = list[df1["ID"]]
    coordinates_dic = get_coordinates_from_csv(fileA)

    #obj.select_points_as_new_agents(obj.req_dic, dict_of_list_map_colors, obj.distance_to_reps_dic)
    obj.closest_agent_distances(obj.coordinates_dic, obj.point_list, list(dict_new_rep.values()))
    obj.get_alpha_dic(obj.distance_to_reps_dic, obj.NR_dic)

    obj.print_result()
    print(len(obj.rep_to_color_dic))
    print("balls (agents) colors: {}".format(obj.rep_to_color_dic))


    # dEquity
    reps =  list(df1['new_cluster'].unique())

    dic_eq = {}
    dic_eq2 = {}
    for att in attributs_list:
        value_list = list(set(df2[att]))
        for val in value_list:
            numerator = min(dic_of_Eq[val], len([i for i in reps if df2[att][i] == val]))
            dic_eq[val] = dic_of_Eq[val] - numerator
            dic_eq2[val] = numerator

    print("sum of all color dic ={}".format(sum(dic_of_Eq.values())))
    print("dEq ={}".format(sum(dic_eq.values())))
    print(sum(dic_eq2.values()))
    print("dEq 2 ={}".format(sum(dic_eq2.values()) / sum(dic_of_Eq.values())))
    end = time.time()
    print(f"Time taken: {(end - start)}seconds")