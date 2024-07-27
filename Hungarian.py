from collections import Counter
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from anytree import Node, RenderTree
import math
from bigtree import Node
from scipy.spatial import cKDTree


class Hungarian:
    def __init__(self, colors_required,df1, df2,nodes,lamda,depth_hierarchy):
        self.colors_required = colors_required # A dictionary with colors as keys and the required number of representatives from each color as values.
        self.df1= df1
        self.df2 = df2
        self.tree_nodes = nodes
        self.agent_most_common_colors = {}
        self.Matrix = list([])
        self.Matrix_org = self.Matrix.copy()
        self.result_Hungarian =[]
        self.lamda = lamda
        self.depth_hierarchy= depth_hierarchy

    def compute_similarity_before_chnge(self):
        # Precompute the powers of lamda
        lamda_powers = [math.pow(self.lamda, i) for i in range(self.depth_hierarchy + 1)]
        df2  = self.df2
        agents = list(df2['group'])
        df1 = self.df1
        dic_agent_color = {}


        for ag in agents:
            dic_agent_color[ag] =0
            color = df1[df1['ID']==ag]['Colors'].iloc[0]
            path = self.find_path_to_root(self.tree_nodes[color])
            path_length = len(path)
            rest = sum(lamda_powers[path_length:self.depth_hierarchy + 1])
            dic_agent_color[ag] += rest

            for p, n in enumerate(path):
                dic_agent_color[ag] += self.RI(ag, n) * lamda_powers[p]

        return list(dic_agent_color.values())
    def find_path_to_root(self,node):
        path = []
        current_node = node
        while current_node is not None:
            path.append(current_node.name)
            current_node = current_node.parent
        return path[::-1]

    def RI(self, group, node):
        if node =='root':
            return 1
        df = self.df2
        list_childern = list(self.tree_nodes[node].parent.children)
        k_x = df[df['group'] == group][node].iloc[0]
        dic_group = df[df['group'] == group]
        print(list_childern)
        list_brothers = list([dic_group[i.name].iloc[0] for i in list_childern])
        if len(list_brothers) ==0:
            return 1
        m= max(list_brothers)
        if max(list_brothers)==0:
            m= 1
        return k_x/m
    def create_community_matrix(self):
        df = self.df2
        # Create a list of colors repeated according to the counts in dictionary A
        colors = []
        for color, count in self.colors_required.items():
            colors.extend([color])

        # Create a set of all agents
        agents = list(df['group'])



        # Initialize an empty matrix
        community_matrix = np.zeros((len(agents), len(colors)))
        df_grouped = df.groupby('group')

        # Precompute the powers of lamda
        lamda_powers = [math.pow(self.lamda, i) for i in range(self.depth_hierarchy + 1)]

        # Iterate over each agent
        for i, agent in enumerate(agents):
            print("i={}, agent ={}".format(i, agent))
            agent_data = df_grouped.get_group(agent)

            # Iterate over each color
            for j, color in enumerate(colors):
                path = self.find_path_to_root(self.tree_nodes[color])
                path_length = len(path)
                rest = sum(lamda_powers[path_length:self.depth_hierarchy + 1])
                community_matrix[i, j] += rest

                for p, n in enumerate(path):
                    community_matrix[i, j] += self.RI(agent, n) * lamda_powers[p]

        skip =0
        community_matrix_copy = community_matrix.copy()
        for c in colors:

            index_c = colors.index(c)
            num_duplicate = self.colors_required[c]-1
            if num_duplicate > 0:
                # Extract the column to duplicate
                column_to_duplicate = community_matrix_copy[:, index_c]
                # Stack the column y times
                duplicated_column = np.column_stack([column_to_duplicate] * num_duplicate)
                # Insert the duplicated column back into the matrix
                community_matrix = np.insert(community_matrix, [skip + 1], duplicated_column, axis=1)
                skip = skip + num_duplicate+1
        self.Matrix_org = np.array(community_matrix)
        return community_matrix

    import numpy as np
    def create_count_matrix(self, matrix):
        # Create a new matrix with the same size as the original matrix
        new_matrix = np.zeros_like(matrix, dtype=int)

        # Flatten and sort the matrix
        sorted_flat_matrix = np.sort(matrix, axis=None)

        # Create a count map to convert matrix values to the correct counts
        count_map = {val: len(sorted_flat_matrix) - np.searchsorted(sorted_flat_matrix, val, side='right') for val in
                     np.unique(matrix)}

        # Use the count map to convert values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 0:
                    new_matrix[i, j] = 3000000
                else:
                    new_matrix[i, j] = count_map[matrix[i, j]]

        self.Matrix = new_matrix
        return new_matrix

    def play_hungarian_algo(self):
        self.Matrix = np.array(self.Matrix)
        row_ind, col_ind = linear_sum_assignment(self.Matrix)
        # Create a list of colors repeated according to the counts in dictionary A
        colors = []
        for color, count in self.colors_required.items():
            colors.extend([color]*count)
        # Create a set of all agents
        agents = list(self.agent_most_common_colors.keys())
        #print("indexes of centers = {}".format(dic_plot))
        list_res = [i for i in self.Matrix_org[row_ind, col_ind]]
       # dic_plot ={agents[i[0]]:colors[i[1]] for i in dict(zip(row_ind, col_ind)).items() }
        self.result_Hungarian = list(list_res)
        return list_res


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
                nodes[node_value] = Node(name=node_value, parent=nodes.get(parent_value, None))
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
def create_hierarchy_uniform(k,csv_file_path, columns):
    # Read data from CSV file
    df = pd.read_csv(csv_file_path,encoding='ISO-8859-1')
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



if __name__ == '__main__':


    fileA = 'Data/POI_radius_1000.csv'
    fileB = 'Data/Point.csv'
    fileC = "Data/POI_group_value_counts_1000_uni.csv"

    attributs_list = ["root", "Colors"]

    df2 = pd.read_csv(fileB,encoding='ISO-8859-1')
    df3 = pd.read_csv(fileC,encoding='ISO-8859-1')
    nodes =create_hierarchy_uniform(  1000,fileB, attributs_list)
    dic_of_absEq,dic_of_Eq  = update_abs_eq(nodes)
    print(dic_of_absEq)

    print(sum([i for i in dic_of_absEq.values()]))
    dic_of_constraints2 ={}
    for i in dic_of_absEq.keys():
        if dic_of_absEq[i]>0:
           dic_of_constraints2[i]=dic_of_absEq[i]
    print("End2")



    obj = Hungarian(dic_of_constraints2,df2, df3,nodes,0.5,1)


    obj.create_community_matrix()
    print("End1")

    obj.create_count_matrix(np.array(obj.Matrix_org))
    print("End")
    list_rep =obj.play_hungarian_algo()

    # Result before
    before_list = obj.compute_similarity_before_chnge()
    print(before_list)
    print("Mean:")
    print("The MaxSum before running Hungarian algo = {}".format(np.mean(before_list)))
    print("The MaxSum after running Hungarian algo = {}".format(np.mean(list_rep)))
    print("Sum:")
    print("The MaxSum before running Hungarian algo = {}".format(np.sum(before_list)))
    print("The MaxSum after running Hungarian algo = {}".format(np.sum(list_rep)))
