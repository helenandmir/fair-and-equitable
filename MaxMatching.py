import sys
import networkx
from collections import defaultdict

sys.setrecursionlimit(50000)


class CR:
    def __init__(self, req_dic, ball_colors_dic, list_of_attributs_dic, balls_dic, distance_to_reps_dic):
        self.ball_colors_dic = ball_colors_dic
        self.req_dic = req_dic
        self.balls_dic = balls_dic
        self.list_of_attributs_dic = list_of_attributs_dic
        self.distance_to_reps_dic = distance_to_reps_dic
        self.list_nodes1 = []
        self.list_nodes2 = []
        self.list_edges = []
        self.dic_new_center = {}

    def add_nodes1(self):
        temp_list = []
        sorted_dict = dict(sorted(self.req_dic.items(), key=lambda item: item[1], reverse=True))
        for color in sorted_dict:
            count = sorted_dict[color]
            # Consistent insertion order
            temp_list.extend([f"{color}{j}" for j in range(int(count))])
        self.list_nodes1 = temp_list  # Order is preserved based on req_dic

    def add_nodes2(self):
        sorted_list_nodes2 = sorted(list(self.ball_colors_dic.keys()), key=lambda x: len(self.ball_colors_dic[x]))
        self.list_nodes2 = sorted_list_nodes2  # Order is preserved based on insertion order

    def add_edges(self):
        sorted_dict = dict(sorted(self.req_dic.items(), key=lambda item: item[1], reverse=True))
        for i in self.list_nodes2:
            for c in sorted_dict:
                if c in self.ball_colors_dic[i]:
                    # Deterministic comparison
                    nodes = [n for n in self.list_nodes1 if c == ''.join(ch for ch in n if not ch.isdigit())]
                    for b in nodes:
                        self.list_edges.append((i, b))

    def extract_color(self, pair_set):
        modified_pairs = []
        # Deterministic iteration over pair_set
        for number, color in pair_set:
            if isinstance(color, int):
                number, color = color, number
            color = ''.join(char for char in str(color)[::-1] if not char.isdigit())[::-1]
            modified_pairs.append((number, color))
        return modified_pairs  # Order depends on the consistent iteration of pair_set

    def replace_rep(self, pair_set):
        new_reps = []
        dic_new_reps = {}

        # Preprocess: Create a mapping from color to set of p
        color_to_ps = defaultdict(set)
        for attr_dict in self.list_of_attributs_dic:
            for p, c in attr_dict.items():
                color_to_ps[c].add(p)

        # Keep track of p's that have been assigned to avoid duplicates
        assigned_ps = set()

        for number, color in pair_set:
            # Get all p's with the desired color
            ps_with_color = color_to_ps.get(color, set())

            # Get p's associated with the current number
            ps_in_number = set(self.balls_dic[number])

            # Available p's are those with the desired color, in the current number, and not already assigned
            available_ps = ps_with_color & ps_in_number - assigned_ps

            if available_ps:
                # Select the p with the minimum distance (with tie-breaker on p)
                new_r = min(available_ps, key=lambda x: (self.distance_to_reps_dic.get(x), x))
                new_reps.append((number, new_r))
                dic_new_reps[new_r] = color
                assigned_ps.add(new_r)
            else:
                print("NOOO")
        return new_reps, dic_new_reps

    def count_colors_pairs(self, reps_color):
        colors_dict = {}
        for _, color in reps_color:
            colors_dict[color] = colors_dict.get(color, 0) + 1
        return colors_dict

    def create_graph(self):
        self.add_nodes1()
        self.add_nodes2()
        self.add_edges()
        G = networkx.Graph()
        G.add_nodes_from(self.list_nodes1)
        G.add_nodes_from(sorted(self.list_nodes2))
        G.add_edges_from(self.list_edges)
        P = networkx.max_weight_matching(G, maxcardinality=True)
        P = self.extract_color(set(P))
        new_reps, dic_new_reps_colors = self.replace_rep(P)
        reps_colors_num = self.count_colors_pairs(P)
        return new_reps, dic_new_reps_colors, reps_colors_num
