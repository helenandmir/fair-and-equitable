
from collections import Counter
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import pandas as pd
import time
import matplotlib.pyplot as plt
from anytree import Node, RenderTree
import math
from bigtree import Node
from scipy.spatial import cKDTree

class Hungarian:
    def __init__(self, colors_required, df1, df2, nodes, lamda, depth_hierarchy):
        self.colors_required = colors_required # A dictionary with colors as keys and the required number of representatives from each color as values.
        self.df1 = df1
        self.df2 = df2
        self.tree_nodes = nodes
        self.agent_most_common_colors = {}
        self.Matrix = []
        self.Matrix_org = self.Matrix.copy()
        self.result_Hungarian = []
        self.lamda = lamda
        self.depth_hierarchy = depth_hierarchy

    def find_path_to_root(self, node):
        path = []
        current_node = node
        while current_node is not None:
            path.append(current_node.name)
            current_node = current_node.parent
        return path[::-1]

    def RI(self, group, node):
        df = self.df2
        list_children = list(self.tree_nodes[node].children)
        k_x = df[df['group'] == group][node].iloc[0]
        dic_group = df[df['group'] == group]
        list_brothers = list([dic_group[i.name].iloc[0] for i in list_children])
        if len(list_brothers) == 0:
            return 1
        m = max(list_brothers)
        if max(list_brothers) == 0:
            m = 1
        return k_x / m

    def create_community_matrix(self):
        df = self.df2
        # Create a list of colors repeated according to the counts in dictionary A
        colors = []
        for color, count in self.colors_required.items():
            colors.extend([color] * count)

        # Create a set of all agents
        agents = list(df['group'])

        # Initialize an empty matrix
        community_matrix = np.zeros((len(agents), len(colors)))

        # Fill the matrix with values
        for i, agent in enumerate(agents):
            for j, color in enumerate(colors):
                community_matrix[i, j] = self.calculate_value(agent, color)
        
        return community_matrix

    def calculate_value(self, agent, color):
        # Implement your logic here
        return np.random.random()  # Example value, replace with actual calculation

    # Other methods can be added here
