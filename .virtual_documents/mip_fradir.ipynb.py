from mip import *
import networkx as nx
import pickle
from itertools import product
import numpy as np
import pandas as pd
import time


# The network
g = nx.read_gml('networks/italy_withLength.gml', label="id")
L = len(g.edges)
links = range(L)


# The cut SRLGs
with open ('min_cut_SRLGs/italy_complete_10-4', 'rb') as fp:
    cut_srlgs = pickle.load(fp)
S = len(cut_srlgs)


# The matrix of the intensity values, dimensions: [L,P,M] (link, position, magnitude)
intensity = np.load('intensities/italy_ds16.npy')


# The matrix of earthquake probabilities, dimensions: [P,M] (position, magnitude)
prob_matrix = pd.read_csv('earthquake_probabilities/italy_ds16.csv').drop(['Lat', 'Long'], axis=1).to_numpy()
P, M = prob_matrix.shape
epicenters = range(P)
magnitudes = range(M)


# Parameters
Hnull = 6
T = 0.0001
cost = 1


# Compressing the problem, to 1 SLRG and the minimum number of earthquakes
cut_srlgs = cut_srlgs[:30]
S = len(cut_srlgs)
print(cut_srlgs)

column_mask = np.full(intensity.shape[2], fill_value=False, dtype=bool)
row_mask = np.full(intensity.shape[1], fill_value=False, dtype=bool)

for srlg in cut_srlgs:
    srlg_mask = np.full(intensity.shape[1:], fill_value=True, dtype=bool)
    for link in srlg:
        idx = list(g.edges).index(link)
        print(idx, link, g.edges[link]['length'])
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


srlg = cut_srlgs[2]
edges = list(g.edges)
link_mask = np.full(intensity.shape[1:], fill_value=True, dtype=bool)
for l in srlg:
    l_idx = edges.index(l)
    link_mask &= intensity[l_idx] > H[l_idx]



# Heuristic 1 & 2

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
        p = get_SRLG_probability_matrix(s, g, intensity, H, prob_matrix).sum()
        if p < threshold:
            active_srlgs.remove(s)
    return active_srlgs

def countSRLGlinks(srlgs, network):
    edges = list(network.edges)
    partofSRLG = np.zeros(len(edges), dtype=int)
    for srlg in srlgs:
        for l in srlg:
            partofSRLG[edges.index(l)] += 1
    #return dict(zip(edges, partofSRLG))
    return partofSRLG

def get_edge_to_improve_1(srlgs, network):
    edges = list(network.edges)
    partofSRLG = countSRLGlinks(srlgs, network)
    max_indexes = [i for i, j in enumerate(partofSRLG) if j == max(partofSRLG)]
    #print(max_indexes)
    if len(max_indexes) > 1:
        max_index = max_indexes[0]
        min_length = network.edges[edges[max_index]]['length']
        for idx in max_indexes:
            length = network.edges[edges[idx]]['length']
            if length < min_length:
                max_index = idx
                min_length = length
        return max_index
    else:
        return max_indexes[0]

def heuristic_1(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
    active_srlgs = remove_improbable_SRLGs(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
    cost = 0
    edges = list(network.edges)
    while len(active_srlgs):
        edge_to_improve = get_edge_to_improve_1(active_srlgs, network)
        cost += network.edges[edges[edge_to_improve]]['length']
        #print(edge_to_improve)
        intensity_tolerance[edge_to_improve] += 1
        active_srlgs = remove_improbable_SRLGs(active_srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
    print(f'H1 Cost: {cost}')

def get_SRLG_probability_reduction(link_idx, srlg, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
    srlg_prob_matrix = get_SRLG_probability_matrix(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix)
    srlg_prob = srlg_prob_matrix.sum()
    intensity_tolerance[link_idx] += 1
    new_srlg_prob_matrix= get_SRLG_probability_matrix(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix)
    new_srlg_prob = new_srlg_prob_matrix.sum()
    intensity_tolerance[link_idx] -= 1
    return srlg_prob - new_srlg_prob

def get_edge_to_improve_2(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
    edges = list(network.edges)
    probability_reduction_values = np.zeros(len(edges))
    for idx, edge in enumerate(edges):
        probability_reduction = 0
        for srlg in srlgs:
            if edge in srlg:
                probability_reduction += get_SRLG_probability_reduction(idx, srlg, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
        probability_reduction_values[idx] = probability_reduction / network.edges[edge]['length']
    return np.argmax(probability_reduction_values)

def heuristic_2(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
    active_srlgs = remove_improbable_SRLGs(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
    cost = 0
    edges = list(network.edges)
    while len(active_srlgs):
        edge_to_improve = get_edge_to_improve_2(active_srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
        cost += network.edges[edges[edge_to_improve]]['length']
        #print(edge_to_improve)
        intensity_tolerance[edge_to_improve] += 1
        active_srlgs = remove_improbable_SRLGs(active_srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
    print(f'H2 Cost: {cost}')


for T in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
    print(f'T: {T}')
    H = np.ones(L) * 6
    heuristic_1(cut_srlgs, g, intensity, H, prob_matrix, T)
    H = np.ones(L) * 6
    heuristic_2(cut_srlgs, g, intensity, H, prob_matrix, T)



# ILP

start = time.perf_counter()

#Model
model = Model(sense=MINIMIZE, solver_name=GRB)

#Variables
deltaH = [model.add_var(var_type=INTEGER, lb=0, ub=6) for l,_ in enumerate(g.edges)]
Z = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in range(S)]
Y = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in links]
print("get_ipython().run_line_magic(".1f", " s:\tVáltozók létrehozva...\"%(time.perf_counter()-start))")


#Objective Function
model.objective = xsum( g.edges[link_id]['length'] * deltaH[link_idx] for link_idx,link_id in enumerate(g.edges) )
print("get_ipython().run_line_magic(".1f", " s:\tCélfüggvény létrehozva...\"%(time.perf_counter()-start))")


#Constraint 1
for l,p,m in product(*[links, epicenters, magnitudes]):
    model.add_constr( Y[l][p][m] >= 1 - ((Hnull + deltaH[l]) / intensity[l,p,m]) )
print("get_ipython().run_line_magic(".1f", " s:\tElső egyenlet létrehozva...\"%(time.perf_counter()-start))")


#Constraint 2
for (c,s), p, m in product(*[enumerate(cut_srlgs), epicenters, magnitudes]):
    model.add_constr( Z[c][p][m] >= (xsum(Y[list(g.edges).index(linkID)][p][m] for linkID in s) - len(s) + 1) )
print("get_ipython().run_line_magic(".1f", " s:\tMásodik egyenlet létrehozva...\"%(time.perf_counter()-start))")

#Constraint 3
for c,_ in enumerate(cut_srlgs):
    model.add_constr(xsum( Z[c][p][m] * prob_matrix[p,m] for p,m in product(epicenters,magnitudes) ) <= T, "c3_"+str(c))
print("get_ipython().run_line_magic(".1f", " s:\tHarmadik egyenlet létrehozva...\"%(time.perf_counter()-start))")

#Start optimization
model.optimize()#max_seconds=
print("get_ipython().run_line_magic(".1f", " s:\tMegoldás megtalálva...\"%(time.perf_counter()-start))")


# The probability of occurance of the SRLGs
for c,s in enumerate(cut_srlgs):
    print(s, sum(Z[c][p][m].x * prob_matrix[p,m] for p,m in product(epicenters,magnitudes) ))


# The result
selected = [(deltaH[l].x,e, g.edges[e]['length']) for l,e in enumerate(g.edges) if deltaH[l].x >= 0.5]
print(*selected, sep='\n')
#print([dH.x for dH in deltaH])


# Add additional constraint
# It does not deletes Constraint 3 but completes it
for c,s in enumerate(cut_srlgs):
    model.add_constr(xsum( Z[c][p][m] * prob_matrix[p,m] for p,m in product(epicenters,magnitudes) ) <= 0.00001, "c3_"+str(c))


# Start optimization
model.optimize()


def ff(a,b):
    return a*2 + b**2


#koltseg =  [11, 3,3, 6, 1, 9, 16,78]
koltseg = 8
fejlesztes = [17,6,4,61,16,42,156, 7]

y = [1,1,0,1,0,1,0,0]
z = [0,1,1,1,0,1,0,1]

items = len(fejlesztes)

m = Model("fradir")

#decision variable
x = [m.add_var(var_type=BINARY) for i in range(items)]
fejlesztes_Dvar = [m.add_var(var_type=INTEGER, lb=1, ub=6) for i in range(items)]

#objective
m.objective = maximize(xsum(koltseg * fejlesztes[i] * x[i] for i in range(items)))

#cons
#m += xsum(y[i] * x[i] for i in range(items)) >= 2

#m += xsum(z[i] * x[i] for i in range(items)) >= 2

#m += xsum(((y[i] + z[i]) % 2) * x[i]  for i in range(items)) == 0

for i in range(items):
    m += 10 * x[i] <= fejlesztes[i]

#m += xsum(( fejlesztes[i] * x[i]  for i in range(items))) <= 14

m += xsum(x[i] for i in range(items)) == 2


m.optimize()

selected = [i for i in range(items) if x[i].x >= 0.99]
print("selected items: {}".format(selected))


ff(3,4)


import random
from gekko import GEKKO
import numpy as np


koltseg =  [11, 3,3, 6, 1, 9, 16,78]
fejlesztes = [17,67,4,61,16,42,156, 7]

y = [1,0,1,1,0,1,1,0]
z = [0,1,1,1,0,1,1,1]

prob = [0.5, 1.2, 1.5, 1.7, 0.1, 1.8, 0.4, 1.9]

items = len(koltseg)

# Create model
m = GEKKO()

# Variables
x = m.Array(m.Var,items,lb=0,ub=1,integer=True)
fejlesztes_Dvar = m.Array(m.Var, items, lb=1,ub=6, integer=True)
b = m.Array(m.Var, items, lb=1,ub=8, integer=True)
#x2 = m.Array(m.Var, len(w),lb=0,ub=1,integer = True)

# Objective
m.Maximize(sum(koltseg[i] * fejlesztes_Dvar[i] * x[i] for i in range(items) ))

# Constraint
m.Equation(sum([(y[i] + z[i]) * x[i] for i in range(items)]) == 2)

m.Equation(sum([x[i] for i in range(items)]) == 1)

m.Equation(sum((z[i] * prob[i]) *x[i] for i in range(items)) <=1.6 )

# Optimize with APOPT
m.options.SOLVER = 1

m.solve(disp = False)

# Print the value of the variables at the optimum
print(x)
for i in range(len(x)):
    if x[i][0] == 1.0:
        print("Sorszam: {}., koltseg: {}, fejlesztes: {}".format(i,koltseg[i], fejlesztes_Dvar[i][0]))





data = np.load('italy.npy')


pd.DataFrame(data[0]).describe()






