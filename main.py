import math
import time
import pandas as pd
from f_FairKCenters import F_FairKCenter
import random
from anytree import Node, RenderTree

from bigtree import Node


def update_abs_eq(list_of_constraints):
    dic_of_constraints = {}
    dic_of_constraints2 = {}

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


def create_hierarchy_random(k, csv_file_path, columns):
    # Dictionary to store created nodes
    nodes = {}

    # Read data from CSV file
    df = pd.read_csv(csv_file_path)
    num_of_points = len(list(df["root"]))

    # Create a new node for "root
    nodes["root"] = Node("root", eq=k, abs_eq=k, abs_eq_temp=k, parent=None)

    for i in columns[1:]:
        list_of_val = list(df[i].unique())
        num_from_att = random.randint(1, len(list_of_val))
        random_items = random.sample(list_of_val, num_from_att)
        k_row = k
        for val in list_of_val:
            parent_column = columns[columns.index(i) - 1]
            parent_value = df[df[i] == val][parent_column].values[0]
            parent_abs_eq_temp = nodes[parent_value].abs_eq_temp
            if val in random_items and parent_abs_eq_temp > 0:
                num_of_point_val = len([j for j in df[i] if j == val])
                val_eq = min(parent_abs_eq_temp, k_row, num_of_point_val)
                rand_val_eq = random.randint(0, val_eq)
                k_row = k_row - rand_val_eq
                nodes[val] = Node(val, eq=rand_val_eq, abs_eq=rand_val_eq, abs_eq_temp=rand_val_eq,
                                  parent=nodes[parent_value])
                nodes[parent_value].abs_eq_temp = nodes[parent_value].abs_eq_temp - rand_val_eq
            else:
                nodes[val] = Node(val, eq=0, abs_eq=0, abs_eq_temp=0, parent=nodes[parent_value])

    return nodes


def create_hierarchy_relative(k, csv_file_path, columns):
    # Read data from CSV file
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
    num_of_points = len(list(df["root"]))

    # Dictionary to store created nodes
    nodes = {}

    # Iterate through each column in the hierarchical order
    for column in columns:
        for value in df[column].unique():
            parent_node = None

            num_of_value = len([i for i in df[column] if i == value])
            eq_num = math.floor((num_of_value / num_of_points) * k)

            # Check if there is a parent in the previous columns
            if columns.index(column) > 0:
                parent_column = columns[columns.index(column) - 1]
                parent_value = df[df[column] == value][parent_column].values[0]
                parent_node = nodes[parent_value]

            # Create a new node
            nodes[value] = Node(value, eq=eq_num, abs_eq=eq_num, parent=parent_node)

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


def create_hierarchy_uniform(k, csv_file_path, columns):
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
            uni_eq = math.floor(parent_eq / val_siblings_num)
            num_of_point_val = len([j for j in df[i] if j == val])
            val_eq = min(uni_eq, num_of_point_val)
            nodes[val] = Node(val, eq=val_eq, abs_eq=val_eq, parent=nodes[parent_value])

    return nodes


if __name__ == '__main__':
    # ------- Runs on real database (POI,Temp,Businesses)
    # # Runs for a real database
    # k = 1000
    # fileA = f'Data/Point_radius_1000.csv'
    # fileB = f'Data/Point.csv'
    #
    # attributs_list = ["root", "Colors"]
    #
    # nodes = create_hierarchy_uniform(k, fileB, attributs_list)
    # dic_of_absEq, dic_of_Eq = update_abs_eq(nodes)
    #
    # dic_of_constraints2 = {}
    # for i in dic_of_absEq.keys():
    #     if dic_of_absEq[i] > 0:
    #         dic_of_constraints2[i] = dic_of_absEq[i]
    #
    # # Algorithm Fair (FairNoEquity)
    # f_2fair = F_FairKCenter(dic_of_Eq,dic_of_absEq, fileA, fileB,attributs_list, k)
    # f_2fair.f()
    #
    #
    # # Algorithm FairEq (f -FairEquitableReps)
    # # f_fair_obj = F_FairKCenter(dic_of_absEq, dic_of_constraints2, fileA, fileB, attributs_list, k)
    # # f_fair_obj.F_fair_main_loop()

    # Trial 1 -number of sensitive attribute values (200,000 data items, k=1000)
    m_values = range(5, 51, 5)
    k = 1000
    for m in m_values:
        print(f"---->Generating dataset for m  = {m}<----")

        fileA = f'trial01/radius_trial01_m_{m}.csv'
        fileB = f'trial01/trial01_m_{m}.csv'

        attributs_list = ["root", "SensitiveAttribute"]  #

        nodes = create_hierarchy_uniform(k, fileB, attributs_list)
        dic_of_absEq, dic_of_Eq = update_abs_eq(nodes)

        dic_of_constraints2 = {}
        for i in dic_of_absEq.keys():
            if dic_of_absEq[i] > 0:
                dic_of_constraints2[i] = dic_of_absEq[i]
        t1 = time.perf_counter(), time.process_time()

        print("->>dic_of_constraints<<-")
        f_fair_obj = F_FairKCenter(dic_of_absEq, dic_of_constraints2, fileA, fileB, attributs_list, k)

        f_fair_obj.F_fair_main_loop()

    #
    #
    # # Trial 2 -numbers of representatives k (200,000 data items, m=15)
    # k_values =range(200, 2001, 200)
    #
    # for k in k_values:
    #     print(f"---->Generating dataset for k  = {k}<----")
    #
    #     fileA = f'trial02/radius_trial02_k_{k}.csv'
    #     fileB = f'trial02/trial02.csv'
    #
    #     attributs_list = ["root", "SensitiveAttribute"]  #
    #
    #     nodes = create_hierarchy_uniform(k, fileB, attributs_list)
    #     dic_of_absEq, dic_of_Eq = update_abs_eq(nodes)
    #
    #     dic_of_constraints2 = {}
    #     for i in dic_of_absEq.keys():
    #         if dic_of_absEq[i] > 0:
    #             dic_of_constraints2[i] = dic_of_absEq[i]
    #     t1 = time.perf_counter(), time.process_time()
    #
    #     print("->>dic_of_constraints<<-")
    #     f_fair_obj = F_FairKCenter(dic_of_absEq, dic_of_constraints2, fileA, fileB, attributs_list, k)
    #
    #     f_fair_obj.F_fair_main_loop()

    # # Trial 3 -different dataset sizes (k=1000, m=15)
    # n_size = range(200000,200001,100000)
    # k=1000
    # for n in n_size:
    #     print(f"---->Generating dataset for n  = {n}<----")
    #
    #     fileA = f'trial03/radius_trial03_size_{n}.csv'
    #     fileB = f'trial02/trial03_size_{n}.csv'
    #
    #     attributs_list = ["root", "SensitiveAttribute"]  #
    #
    #     nodes = create_hierarchy_uniform(k, fileB, attributs_list)
    #     dic_of_absEq, dic_of_Eq = update_abs_eq(nodes)
    #
    #     dic_of_constraints2 = {}
    #     for i in dic_of_absEq.keys():
    #         if dic_of_absEq[i] > 0:
    #             dic_of_constraints2[i] = dic_of_absEq[i]
    #     t1 = time.perf_counter(), time.process_time()
    #
    #     print("->>dic_of_constraints<<-")
    #     f_fair_obj = F_FairKCenter(dic_of_absEq, dic_of_constraints2, fileA, fileB, attributs_list, k)
    #
    #     f_fair_obj.F_fair_main_loop()
