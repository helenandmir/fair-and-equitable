import sys
import time
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.pyplot
import random
from matplotlib import pyplot as plt, colors
from math import sin, cos, sqrt, atan2
import geopy.distance
import pandas as pd
import csv
import os

script_dir = os.path.dirname(__file__)

fileA =os.path.join(script_dir, '..', 'Data', 'Temp.csv')
fileB =os.path.join(script_dir, '..', 'Data', 'Temp_radius_1000.csv')
class CreateDataTable:
    def __init__(self,df_from,col_from,df_to,k):
        self.dic_id_NR = {}  # A dictionary that holds the neighborhood radius for each point

        self.df1 = pd.read_csv(df_from, usecols=col_from)
        self.colors_list =list(set(self.df1.Continent))+list(set(self.df1.Country))+list(set(self.df1.City)) #list(set(self.df1.INDUSTRY))+list(set(self.df1.general_INDUSTRY))


        self.df2 = pd.read_csv(df_to)

        self.K = k #number of centers


    def initialization_dic_loc_and_dis(self):
        """
        initialization dic_id_loc and dic_id_dis dictionary by k-d tree
        """
        print("initialization")
        all_point = list(self.df1.ID)
        N = len(all_point)

        balance = np.ceil(N / self.K)
        X_list = np.array(self.df1.X)
        Y_list = np.array(self.df1.Y)
        Z_list = np.array(self.df1.Z)
        tuple_coordinates = tuple(zip(X_list, Y_list,Z_list))
        arr_coordinates = np.array(list(tuple_coordinates))
        tree = KDTree(arr_coordinates)

        for id in all_point:
            print(id)
            dist, ind = tree.query([[self.df1.X[id],self.df1.Y[id],self.df1.Z[id]]], int(balance)-1)
            #self.dic_id_loc[id] = [self.df1.Longitude[id], self.df1.Latitude[id]]
            #self.dic_id_dis[id] = dist.tolist()[0]

            #ret_dis = self.convert_euclidean(id, ind[0])
            # list_NR.append(min(dist_c[0]))
            self.dic_id_NR[id] = max(dist[0])

            #self.initialization_NR_type_one(id,list_dis)



        #if self.radius_type == 2:
         #  self.initialization_NR_type_two()

        self.df2["NR_TYPE_ONE"] = self.dic_id_NR.values()
        self.df2.to_csv(fileB, index=False)
        self.df2.head()

    def writing_to_table(self):
        print("in writing")



        tree_colors = {}
        dic_list_colors = {}

        for c in self.colors_list:
            print(c)
            #list_color = [i for i in self.df1.ID if self.df1.INDUSTRY[i] == c or self.df1.general_INDUSTRY[i] == c ]
            list_color = [i for i in self.df1.ID if self.df1.City[i] == c or self.df1.Country[i] == c or self.df1.Continent[i] == c]


            tuple_color = tuple(zip(list(self.df1.X[list_color]), list(self.df1.Y[list_color]),list(self.df1.Z[list_color])))

            tree_c = KDTree(np.array(list(tuple_color)))
            tree_colors[c] = tree_c
            dic_list_colors[c] = list_color

        for c in self.colors_list:
            print(c)
            list_delet_color = []
            list_close_index =[]
            list_close_dis =[]
            for id in self.df1.ID:
                print(id)
                dist_c, ind_c = tree_colors[c].query([[self.df1.X[id], self.df1.Y[id],self.df1.Z[id]]], 1)

                ind =dic_list_colors[c][ind_c[0][0]]
                list_close_index.append(ind)
                list_close_dis.append(dist_c[0][0])

            # self.df3[str(c)] = list_close_index
            # self.df3.to_csv("save_data_ind.csv")
            # self.df3.head()

            self.df2[str(c)] = list_close_dis
            self.df2.to_csv(fileB)
            self.df2.head()
def main():
    file = open(fileA)

    type(file)
    csvreader = csv.reader(file)
    header = next(csvreader)
    list_header = list(header)
    t = CreateDataTable(fileA, ["ID", "X", "Y", "Z","root", "Continent", "Country", "City"], fileB, 1000)
    t.initialization_dic_loc_and_dis()
    t.writing_to_table()


if __name__ == '__main__':
    main()