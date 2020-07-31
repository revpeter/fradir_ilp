from mip import *
import networkx as nx
import pickle
from itertools import product
import numpy as np
import pandas as pd
import time


# The network
g = nx.read_gml('networks/italy.gml', label="id")
L = len(g.edges)
links = range(L)


# The cut SRLGs
with open ('min_cut_SRLGs/italy_corrected', 'rb') as fp:
    cut_srlgs = pickle.load(fp)
S = len(cut_srlgs)


# The matrix of the intensity values, dimensions: [L,P,M] (link, position, magnitude)
intensity = np.load('intensities/italy.npy')


# The matrix of earthquake probabilities, dimensions: [P,M] (position, magnitude)
prob_matrix = pd.read_csv('earthquake_probabilities/italy.csv').drop(['Lat', 'Long'], axis=1).to_numpy()
P, M = prob_matrix.shape
epicenters = range(P)
magnitudes = range(M)


# Parameters
Hnull = 6
T = 0.0001
cost = 1


# Compressing the problem, to 1 SLRG and the minimum number of earthquakes
cut_srlgs = cut_srlgs[2:3]
S = len(cut_srlgs)
print(cut_srlgs)

column_mask = np.full(intensity.shape[2], fill_value=False, dtype=bool)
row_mask = np.full(intensity.shape[1], fill_value=False, dtype=bool)

for srlg in cut_srlgs:
    srlg_mask = np.full(intensity.shape[1:], fill_value=True, dtype=bool)
    for link in srlg:
        idx = list(g.edges).index(link)
        print(idx, link)
        link_mask = intensity[idx]>6
        srlg_mask &= link_mask
    column_mask |= srlg_mask.any(axis=0)
    row_mask |= srlg_mask.any(axis=1)

intensity = intensity[:,:,column_mask][:,row_mask]
prob_matrix = prob_matrix[:,column_mask][row_mask]

P, M = prob_matrix.shape
epicenters = range(P)
magnitudes = range(M)

print(f'The shape of the intensity matrix: {intensity.shape}')

start = time.perf_counter()

#Model
model = Model(sense=MINIMIZE, solver_name=GRB)
print("%.1f s:\tModel created..."%(time.perf_counter()-start))

#Variables
deltaH = [model.add_var(var_type=INTEGER, lb=0, ub=6) for l,_ in enumerate(g.edges)]
Z = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in range(S)]
Y = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in links]
print("%.1f s:\tVariables created..."%(time.perf_counter()-start))


#Objective Function
model.objective = xsum( cost * deltaH[l] for l in links )
print("%.1f s:\tObjecive function created..."%(time.perf_counter()-start))


#Constraint 1
for l,p,m in product(*[links, epicenters, magnitudes]):
    model.add_constr( Y[l][p][m] >= 1 - ((Hnull + deltaH[l]) / intensity[l,p,m]) )
print("%.1f s:\tFirst constraint created..."%(time.perf_counter()-start))


#Constraint 2
for (c,s), p, m in product(*[enumerate(cut_srlgs), epicenters, magnitudes]):
    model.add_constr( Z[c][p][m] >= (xsum(Y[list(g.edges).index(linkID)][p][m] for linkID in s) - len(s) + 1) )
print("%.1f s:\tSecond constraint created..."%(time.perf_counter()-start))

#Constraint 3
for c,_ in enumerate(cut_srlgs):
    model.add_constr(xsum( Z[c][p][m] * prob_matrix[p,m] for p,m in product(epicenters,magnitudes) ) <= T, "c3_"+str(c))
print("%.1f s:\tThird constraint created..."%(time.perf_counter()-start))

#Start optimization
model.optimize()#max_seconds=

selected = [(deltaH[l].x,e) for l,e in enumerate(g.edges) if deltaH[l].x >= 0.5]
print("selected items: {}".format(selected))