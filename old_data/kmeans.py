import numpy as np
import matplotlib.pyplot as plt
import random as rnd


def tmp_centre( k, data):
     #print(len(data))
     tmpCentre = rnd.sample(data, k)
     return tmpCentre

def euclidian_distance(data1, data2):
     #print(data1[0])
     #print(data1 ,"\n", data2)
     x1 = np.array([data1[2], data1[3]])
     x2 = np.array([data2[len(data2) - 2],data2[len(data2) - 1]])
     #print("euclidian", np.sqrt(np.sum((x1 - x2)**2)))
     return np.sqrt(np.sum((x1 - x2)**2))

def cluster_association(data, centres, k):
     tab = []
     for i in range(len(data)):
          distance = []
          for j in range(k):
               distance.append(euclidian_distance(data[i], centres[j]))
          tab.append(distance.index(min(distance)))
     return tab

def definitive_centers(data, centres, k):
     tab = []
     for i in range(k):
          new_tab = []
          for j in range(len(data)):
               if centres[j] == i:
                    new_tab.append([int(data[j][2]), int(data[j][3])])
          # print(new_tab)
          tab.append(np.mean(new_tab, axis = 0))
     return tab

def calcul_sse(k,cluster,centre,data):
     sse = []
     for i in range(k):
        tmp = 0
        for j in range(len(data)):
          if(cluster[j] == i):
               tmp += (euclidian_distance(data[j],centre[i]))
          sse.append(tmp)
     return sum(sse) / k