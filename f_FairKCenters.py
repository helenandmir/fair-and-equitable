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
import MaxMatching
from collections import defaultdict
from scipy.spatial import cKDTree
from collections import Counter
import statistics
from FairKCenterEexpanded import FairKCenter


class F_FairKCenter:
    def __init__(self, equity_constraint,req_dic, input_fileA,input_fileB,attributs_list, k):
        df = pd.read_csv(input_fileB,encoding='ISO-8859-1')
        self.fileA = input_fileA # Information about the points, including coordinates and sensitive attributes (color)
        self.fileB = input_fileB # The neighborhood radius of each point
        self.req_colors_dic = req_dic #abs_eq
        self.equity_constraint= equity_constraint #eq
        self.attributs_list = attributs_list

        self.k = k # Number of reps

    def f(self):
        start = time.time()
        obj = FairKCenter(self.req_colors_dic, self.fileA, self.fileB, self.attributs_list, self.k, 0)
        low = 1
        high = 2
        flag =0
        while flag ==0:
            mid = (low+high)/2
            reps = obj.two_fair(2.2)
            print(1.8)
            print(len(reps))
            if len(reps)<=505  and len(reps)>=445:
                break
            elif len(reps)<505:
                high =mid
            else:
                low=mid
        obj.closest_agent_distances(obj.coordinates_dic, obj.point_list, reps)
        obj.get_alpha_dic(obj.distance_to_reps_dic, obj.NR_dic)
        print("reps:")
        print(reps)
        obj.print_result()
        print("numOfReps ={}".format(len(reps)))
        df = pd.read_csv(self.fileB)
        dic_eq = {}
        dic_eq2 ={}
        for att in self.attributs_list:
            value_list = list(set(df[att]))
            for val in value_list:
                numerator =min(self.equity_constraint[val],len([i for i in reps if df[att][i]==val]))
                dic_eq[val] = self.equity_constraint[val] -numerator
                dic_eq2[val] =numerator

        print("sum of all color dic ={}".format(sum(self.equity_constraint.values())))
        print("dEq ={}".format(sum(dic_eq.values())))
        print(sum(dic_eq2.values()))
        print("dEq 2 ={}".format(sum(dic_eq2.values())/sum(self.equity_constraint.values())))
        end = time.time()

        print("**")


    # the main part of the algorithm
    def F_fair_main_loop(self):
        start = time.time()
        for f in range(0, len(self.req_colors_dic) + 1):
            obj = FairKCenter(self.req_colors_dic, self.fileA, self.fileB, self.attributs_list, self.k, f)
            reps = obj.two_fair(1) # get a set of reps from 2FairAlg
            C = MaxMatching.CR(obj.req_dic, obj.balls_colors_dic,obj.list_of_attributs_dic, obj.balls_dic)
            new_reps, dic_new_reps_colors, num_colors_dic = C.create_graph()
            if len(reps) == len(new_reps):
                print(True)
                break
            else:
                print(False)
        print("out of loop!!")
        print("f = {}".format(f))
        obj.rep_to_color_dic.update(dic_new_reps_colors)
        #obj.req_dic = obj.subtract_dicts(obj.req_dic, num_colors_dic)
        obj.req_dic = obj.updatr_dict(dic_new_reps_colors.keys(),obj.req_dic)
        dict_of_list_map_colors = obj.color_points_mapping(obj.req_dic)
        obj.select_points_as_new_agents(obj.req_dic, dict_of_list_map_colors, obj.distance_to_reps_dic)
        obj.closest_agent_distances(obj.coordinates_dic, obj.point_list, list(obj.rep_to_color_dic.keys()))
        obj.get_alpha_dic(obj.distance_to_reps_dic, obj.NR_dic)

        obj.print_result()
        print(len(obj.rep_to_color_dic))
        print("balls (agents) colors: {}".format(obj.rep_to_color_dic))
        print("balls (agents) colors: {}".format(obj.rep_to_color_dic.keys()))
        #obj.plot_point()
        end = time.time()
        print(f"Time taken: {(end - start)}seconds")
        invert_dictionary = obj.invert_dictionary(obj.closest_rep_dic)
        obj.count_unique_values_in_groups(invert_dictionary)
        return self.req_colors_dic,obj.closest_rep_dic,obj.list_of_attributs_dic, obj.distance_to_reps_dic