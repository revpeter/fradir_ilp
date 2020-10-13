from mip import *
import networkx as nx
import pickle
from itertools import product
import numpy as np
import pandas as pd
import time
import resource

def get_SRLG_probability_matrix(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix):
    edges = list(network.edges)
    srlg_occur = np.full(intensity_matrix.shape[1:], fill_value=True, dtype=bool)
    for l in srlg:
        l_idx = edges.index(l)
        srlg_occur &= intensity[l_idx] > intensity_tolerance[l_idx]
    return probability_matrix[srlg_occur]

def remove_improbable_SRLGs(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
    active_srlgs = srlgs.copy()
    for s in active_srlgs.copy():
        p = get_SRLG_probability_matrix(s, g, intensity, intensity_tolerance, prob_matrix).sum()
        if p < threshold:
            active_srlgs.remove(s)
    return active_srlgs


# The network
g = nx.read_gml('networks/italy_995.gml', label="id")
L = len(g.edges)
links = range(L)


# The cut SRLGs
with open ('min_cut_SRLGs/italy_995_10-4', 'rb') as fp:
    cut_srlgs = pickle.load(fp)

active_srlgs = cut_srlgs
S = len(active_srlgs)


# The matrix of the intensity values, dimensions: [L,P,M] (link, position, magnitude)
intensity = np.load('intensities/italy_ds16.npy')


# The matrix of earthquake probabilities, dimensions: [P,M] (position, magnitude)
prob_matrix = pd.read_csv('earthquake_probabilities/italy_ds16.csv').drop(['Lat', 'Long'], axis=1).to_numpy()
P, M = prob_matrix.shape
epicenters = range(P)
magnitudes = range(M)


# Parameters
Hnull = 6
cost = 1
T = 0.01

active_srlgs = remove_improbable_SRLGs(cut_srlgs, g, intensity, np.ones(L) * Hnull, prob_matrix, T)
S = len(active_srlgs)


print(f'The shape of the intensity matrix: {intensity.shape}')
print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

start = time.perf_counter()

#Model
model = Model(sense=MINIMIZE, solver_name=GRB)

#Variables
deltaH = [model.add_var(var_type=INTEGER, lb=0, ub=6) for l,_ in enumerate(g.edges)]
Z = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in range(S)]
Y = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in links]
print("%.1f s:\tVáltozók létrehozva..."%(time.perf_counter()-start))
print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')


#Objective Function
model.objective = xsum( g.edges[link_id]['length'] * deltaH[link_idx] for link_idx,link_id in enumerate(g.edges) )
print("%.1f s:\tCélfüggvény létrehozva..."%(time.perf_counter()-start))
print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')


#Constraint 1
for l,p,m in product(*[links, epicenters, magnitudes]):
    model.add_constr( Y[l][p][m] >= 1 - ((Hnull + deltaH[l]) / intensity[l,p,m]) )
print("%.1f s:\tElső egyenlet létrehozva..."%(time.perf_counter()-start))
print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')


#Constraint 2
for (c,s), p, m in product(*[enumerate(active_srlgs), epicenters, magnitudes]):
    model.add_constr( Z[c][p][m] >= (xsum(Y[list(g.edges).index(linkID)][p][m] for linkID in s) - len(s) + 1) )
print("%.1f s:\tMásodik egyenlet létrehozva..."%(time.perf_counter()-start))
print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

#Constraint 3
for c,_ in enumerate(active_srlgs):
    model.add_constr(xsum( Z[c][p][m] * prob_matrix[p,m] for p,m in product(epicenters,magnitudes) ) <= T, "c3_"+str(c))
print("%.1f s:\tHarmadik egyenlet létrehozva..."%(time.perf_counter()-start))
print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

#Start optimization
model.optimize()#max_seconds=
runtime_ILP = time.perf_counter()-start
print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

# The result
cost = 0
selected = []
upgrade_ILP = []
for l,e in enumerate(g.edges):
    if deltaH[l].x >= 0.5:
        selected.append((deltaH[l].x,e, g.edges[e]['length']))
        upgrade_ILP.append((e,deltaH[l].x))
        cost += deltaH[l].x * g.edges[e]['length']
print(cost)
print(*selected, sep='\n')


# Saving the result
df_cost = pd.read_csv('results/Heuristic_comparison.csv')
with open ('results/Heuristic_upgraded_edges', 'rb') as fp:
    result_edge = pickle.load(fp)

idx = 0
df_cost.loc[idx,'Runtime ILP'] = runtime_ILP
df_cost.loc[idx,'Cost ILP'] = cost
df_cost.to_csv('results/Heuristic_comparison.csv', index=False, float_format='%.5f')
result_edge[idx]['Upgrade ILP'] = upgrade_ILP

df_cost.to_csv('results/Heuristic_comparison.csv', index=False, float_format='%.5f')
with open('results/Heuristic_upgraded_edges', 'wb') as fp:
    pickle.dump(result_edge, fp)