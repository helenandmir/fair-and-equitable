import pandas as pd
import sys
import networkx
from networkx.algorithms.mis import maximal_independent_set
sys.setrecursionlimit(50000)
class CR:
    def __init__(self, req_dic,ball_colors_dic,list_of_attributs_dic,balls_dic):
        self.ball_colors_dic = ball_colors_dic
        self.req_dic =req_dic
        self.balls_dic =balls_dic
        self.list_of_attributs_dic = list_of_attributs_dic
        self.list_nodes1 =[]
        self.list_nodes2 =[]

        self.list_edges = []
        self.dic_new_center={}

    def add_nodes1(self):
        max_length = max(map(int, self.req_dic.values()))  # Finding the maximum length of the list represented by the maximum value in the dictionary
        temp_list = []  # Creating a temporary list to store the content
        for color, count in self.req_dic.items():  # Iterating over each value in the dictionary
            temp_list.extend(
                [color + str(j) for j in range(int(count))])  # Adding the corresponding string to the temporary list
        self.list_nodes1 += temp_list  # Appending the temporary list to the main list


    def add_nodes2(self):
        self.list_nodes2.extend(self.ball_colors_dic)


    def add_edegs(self):
        for i in self.list_nodes2:
            for c in self.ball_colors_dic[i]:
                nodes = [n for n in self.list_nodes1 if c == "".join([i for i in n if not i.isdigit()])]
                self.list_edges.extend((i,b) for b in nodes)


    # Extracts the color part from the second element of each pair in the given set of pairs.
    def extract_color(self, pair_set):
        modified_pairs = []
        for number, color in pair_set:
            if isinstance(color, int):
                number, color = color, number
            color = ''.join(char for char in color[::-1] if not char.isdigit())[::-1]
            modified_pairs.append((number, color))
        return modified_pairs

    def replace_rep(self,pair_set):
        new_reps =[]
        dic_new_reps={}
        for number, color in pair_set:
            for p in self.balls_dic[number]:
                if color in [i[p] for i in self.list_of_attributs_dic] and p not in dic_new_reps :#
                    new_reps.append((number, p))
                    dic_new_reps[p]=color
                    break
        return new_reps,dic_new_reps

    # Counts the number of elements (color) in a set of pairs (reps_color) where each pair consists of a number and a color.
    def count_colors_pairs(self,reps_color):
        colors_dict = {}

        for pair in reps_color:
            color = pair[1]
            colors_dict[color] = colors_dict.setdefault(color, 0) + 1

        return colors_dict

    def create_graph(self):
        self.add_nodes1()
        self.add_nodes2()
        self.add_edegs()
        G = networkx.Graph()
        G.add_edges_from(self.list_edges)
        P=networkx.max_weight_matching(G)
        #print(P)
        P = self.extract_color(set(P))
        #print(P)
        new_reps,dic_new_reps_colors = self.replace_rep(P)
        #print(new_reps)
        reps_colors_num = self.count_colors_pairs(P)
        #print(reps_colors_num)
        return new_reps,dic_new_reps_colors,reps_colors_num

