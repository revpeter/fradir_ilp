from mip import *
from backend import *
from itertools import product
import time

network_name = 'italy_995'
spine_bonus = 0

# The network
g = nx.read_gml(f'networks/{network_name}.gml', label="id")
L = len(g.edges)
links = range(L)


# The cut SRLGs
with open (f'min_cut_SRLGs/{network_name}_2-4', 'rb') as fp:
    cut_srlgs = pickle.load(fp)
S = len(cut_srlgs)


# All SRLG
all_srlgs, _ = get_SRLGs(f'PSRLGs/{network_name}.xml')


# The matrix of the intensity values, dimensions: [L,P,M] (link, position, magnitude)
intensity = np.load(f'intensities/{network_name}_ds16.npy')


# The matrix of earthquake probabilities, dimensions: [P,M] (position, magnitude)
prob_matrix = pd.read_csv('earthquake_probabilities/italy_ds16.csv').drop(['Lat', 'Long'], axis=1).to_numpy()
P, M = prob_matrix.shape
epicenters = range(P)
magnitudes = range(M)

Hnull = 6
cost = 1
TFA = 0.01

start = time.perf_counter()
H = np.array([Hnull+spine_bonus*g.edges[e]['onspine'] for e in g.edges])
#Model
model = Model(sense=MINIMIZE, solver_name=GRB)

#Variables
deltaH = [model.add_var(var_type=INTEGER, lb=0, ub=3) for l,_ in enumerate(g.edges)]
Z = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in range(S)]
Y = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in links]
W = [[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters]
print("%.1f s:\tVáltozók létrehozva..."%(time.perf_counter()-start))


#Objective Function
model.objective = xsum( g.edges[link_id]['length'] * deltaH[link_idx] for link_idx,link_id in enumerate(g.edges) )
print("%.1f s:\tCélfüggvény létrehozva..."%(time.perf_counter()-start))

#Constraint 1
for l,p,m in product(*[links, epicenters, magnitudes]):
    model.add_constr( Y[l][p][m] >= 1 - ((H[l] + deltaH[l]) / intensity[l,p,m]) )
print("%.1f s:\tElső egyenlet létrehozva..."%(time.perf_counter()-start))

#Constraint 2
for (c,s), p, m in product(*[enumerate(cut_srlgs), epicenters, magnitudes]):
    model.add_constr( Z[c][p][m] >= (xsum(Y[list(g.edges).index(linkID)][p][m] for linkID in s) - len(s) + 1) )
print("%.1f s:\tMásodik egyenlet létrehozva..."%(time.perf_counter()-start))

#Constraint 3
for (c,s), p, m in product(*[enumerate(cut_srlgs), epicenters, magnitudes]):
    model.add_constr( W[p][m] >= Z[c][p][m] )
print("%.1f s:\tHarmadik egyenlet létrehozva..."%(time.perf_counter()-start))

model.add_constr(xsum( W[p][m] * prob_matrix[p,m] for p,m in product(epicenters,magnitudes) ) <= TFA )
print("%.1f s:\tNegyedik egyenlet létrehozva..."%(time.perf_counter()-start))

model.write(f'results/{network_name}/{network_name}_SB{spine_bonus}.lp')