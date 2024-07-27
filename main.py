import math
import sys
import time
import csv
from collections import defaultdict
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import random
from scipy.spatial import KDTree
from collections import defaultdict
from scipy.spatial import cKDTree
from collections import Counter
import statistics
from FairKCenterEexpanded import FairKCenter
from f_FairKCenters import F_FairKCenter
import random
from anytree import Node, RenderTree

from bigtree import Node


# def get_new_agents(balls_colors, points_to_old_agents, dis_between_points_to_agents, points_to_colors):
#     new_agent_after_max_sum = {}
#     new_agent_to_colors= {}
#
#     for agent_id in balls_colors:
#         agent_color = balls_colors[agent_id]
#
#         # Step 1: Create a list of points associated with agent_id in A
#         agent_points = [point_id for point_id, rep_id in points_to_old_agents.items() if rep_id == agent_id]
#
#         # Step 2: Remove points with different color from agent_color in C
#         agent_points = [point_id for point_id in agent_points if points_to_colors[point_id] == agent_color]
#
#         # Step 3: Find the point with the smallest value in B
#         try:
#            min_distance_point = min(agent_points, key=lambda point_id: dis_between_points_to_agents[point_id])
#            # Step 4: Add the agent_id with the min_distance_point to new_agent_after_max_sum
#            new_agent_after_max_sum[agent_id] = min_distance_point
#            new_agent_to_colors[min_distance_point] = agent_color
#         except:
#             print("An exception occurred")
#             print("agent_id = {}, ball_color ={}, agent_color={}".format(agent_id,agent_color,points_to_colors[agent_id]))
#
#
#     return new_agent_after_max_sum,new_agent_to_colors

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

# def initialization_nodes(k,file,attributs_list):
#     list_of_nodes =[]
#     df = pd.read_csv(file)
#     num_of_points = len(list(df["root"]))
#     index_node =0
#     for att in attributs_list:
#         list_of_value = list(set(df[att]))
#         index_att = attributs_list.index(att)
#         for v in list_of_value:
#            index_node= index_node+1
#            num_of_v =  len([i for i in list(df[att]) if i == v])
#            eq_num = math.floor((num_of_v/num_of_points)*k)
#            att_parent = attributs_list[index_att-1]
#            if v != 'root':
#                parent_of_v = df[att_parent][list(df[att]).index(v)]
#                list_of_nodes.append(Node(v,eq = eq_num,abs_eq =eq_num,parent=parent_of_v))
#            else:
#                list_of_nodes.append(Node(v,eq = eq_num,abs_eq =eq_num))

def create_hierarchy_random(k,csv_file_path, columns):
    # Dictionary to store created nodes
    nodes = {}

    # Read data from CSV file
    df = pd.read_csv(csv_file_path)
    num_of_points = len(list(df["root"]))

    # Create a new node for "root
    nodes["root"] = Node("root", eq=k, abs_eq=k,abs_eq_temp=k, parent=None)


    for i in columns[1:]:
        list_of_val = list(df[i].unique())
        num_from_att = random.randint(1,len(list_of_val))
        random_items = random.sample(list_of_val, num_from_att)
        k_row =k
        for val in list_of_val:
            parent_column = columns[columns.index(i) - 1]
            parent_value = df[df[i] == val][parent_column].values[0]
            parent_abs_eq_temp = nodes[parent_value].abs_eq_temp
            if val in random_items and parent_abs_eq_temp>0:
                num_of_point_val = len([j for j in df[i] if j == val])
                val_eq = min(parent_abs_eq_temp, k_row, num_of_point_val)
                rand_val_eq = random.randint(0, val_eq)
                k_row = k_row - rand_val_eq
                nodes[val] = Node(val, eq=rand_val_eq, abs_eq=rand_val_eq,abs_eq_temp=rand_val_eq, parent=nodes[parent_value])
                nodes[parent_value].abs_eq_temp = nodes[parent_value].abs_eq_temp -rand_val_eq
            else:
                nodes[val] = Node(val, eq=0, abs_eq=0,abs_eq_temp=0, parent=nodes[parent_value])


    return nodes
def create_hierarchy_relative(k,csv_file_path, columns):
    # Read data from CSV file
    df = pd.read_csv(csv_file_path,encoding='ISO-8859-1')
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


# Function to create the tree

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

if __name__ == '__main__':
    # file_temp = "Data/Try.csv"
    # nodes = create_hierarchy_uniform(10, file_temp, ["root", "continent", "country", "city"])
    # dic_of_absEq, dic_of_Eq = update_abs_eq(nodes)
    # print(dic_of_absEq)
    # print(dic_of_Eq)
    # print(sum([i for i in dic_of_absEq.values()]))
    # print("**")

    fileA = 'Data/Point_radius_500.csv'
    fileB = 'Data/Point.csv'

    attributs_list = ["root", "Colors"]


    nodes =create_hierarchy_relative(  500,fileB, attributs_list)
    dic_of_absEq,dic_of_Eq  = update_abs_eq(nodes)
    print(dic_of_absEq)

    print(sum([i for i in dic_of_absEq.values()]))
    dic_of_constraints2 ={}
    for i in dic_of_absEq.keys():
        if dic_of_absEq[i]>0:
           dic_of_constraints2[i]=dic_of_absEq[i]
    t1 = time.perf_counter(), time.process_time()
    #The initialization of the hierarchical constraints
    # level 0
    # root = Node("root", eq=1000, abs_eq =1000)
    # list_of_constraints = []
    # #level 1
    # Entertainment_and_Leisure = Node("Entertainment and Leisure", eq = 60, abs_eq = 60, parent=root)
    # list_of_constraints.append(Entertainment_and_Leisure)
    # Hospitality_and_Food_Services = Node("Hospitality and Food Services", eq =30 , abs_eq =30, parent=root)
    # list_of_constraints.append(Hospitality_and_Food_Services)
    # Specialty_Services = Node("Specialty Services", eq =10 , abs_eq =10, parent=root)
    # list_of_constraints.append(Specialty_Services)
    # Service_Providers = Node("Service Providers", eq =200 , abs_eq =200, parent=root)
    # list_of_constraints.append(Service_Providers)
    # Retail_and_Sales = Node("Retail and Sales", eq =500 , abs_eq =500, parent=root)
    # list_of_constraints.append(Retail_and_Sales)
    #
    # #level 2
    # Amusement_Arcade = Node("Amusement Arcade", eq=5, abs_eq=5, parent=Entertainment_and_Leisure)
    # list_of_constraints.append(Amusement_Arcade)
    # Amusement_Device_Portable = Node("Amusement Device Portable", eq =20 , abs_eq =20, parent=Entertainment_and_Leisure)
    # list_of_constraints.append(Amusement_Device_Portable)
    # Cabaret = Node("Cabaret", eq=10, abs_eq=10, parent=Entertainment_and_Leisure)
    # list_of_constraints.append(Cabaret)
    # Games_of_Chance = Node("Games of Chance", eq=5, abs_eq=5, parent=Entertainment_and_Leisure)
    # list_of_constraints.append(Games_of_Chance)
    # Cigarette_Retail_Dealer = Node("Cigarette Retail Dealer", eq=200, abs_eq=200, parent=Retail_and_Sales)
    # list_of_constraints.append(Cigarette_Retail_Dealer)
    # Electronics_Store = Node("Electronics Store", eq=50, abs_eq=50, parent=Retail_and_Sales)
    # list_of_constraints.append(Electronics_Store)
    # Electronic_and_Appliance_Service =  Node("Electronic & Appliance Service", eq=100, abs_eq=100, parent=Service_Providers)
    # list_of_constraints.append(Electronic_and_Appliance_Service)
    # Booting_Company = Node("Booting Company", eq=2, abs_eq=2, parent=Specialty_Services)
    # list_of_constraints.append(Booting_Company)
    # Calculate the absolute equity for each value
    # dic_of_constraints = update_abs_eq(list_of_constraints)
    # print("{}.abs_eq = {}".format(root.name, root.abs_eq))
    # for i in list_of_constraints:
    #     print("{}.abs_eq = {}".format(i.name,i.abs_eq))
    print("->>dic_of_constraints<<-")

    # -->>>f_fair_K_center algorithm<<<-
    print("-->>>f_fair_K_center algorithm<<<-")
    start = time.time()
    f_2fair = F_FairKCenter(dic_of_Eq,dic_of_absEq, fileA, fileB,attributs_list, 500)
    f_2fair.f()
    end = time.time()
    print(f"Time taken: {(end - start)}seconds")
    # f_fair_obj = F_FairKCenter(dic_of_Eq,dic_of_constraints2, fileA, fileB,attributs_list, 500)
    #
    # f_fair_obj.F_fair_main_loop()
