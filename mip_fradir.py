from mip import *
from backend import *
from itertools import product
import time
import resource


network_name = 'usa_99'

# The network
g = nx.read_gml(f'networks/{network_name}.gml', label="id")
L = len(g.edges)
links = range(L)


# The cut SRLGs
with open (f'min_cut_SRLGs/{network_name}', 'rb') as fp:
    cut_srlgs = pickle.load(fp)
S = len(cut_srlgs)

# All SRLG
all_srlgs, _ = get_SRLGs('PSRLGs/usa_99_complete_it6.xml')


# The matrix of the intensity values, dimensions: [L,P,M] (link, position, magnitude)
intensity = np.load(f'intensities/{network_name}_ds23.npy')


# The matrix of earthquake probabilities, dimensions: [P,M] (position, magnitude)
prob_matrix = pd.read_csv('earthquake_probabilities/usa_ds23.csv').drop(['Lat', 'Long'], axis=1).to_numpy()
P, M = prob_matrix.shape
epicenters = range(P)
magnitudes = range(M)



# Parameters
Hnull = 6
cost = 1
Ts = [0.01, 0.005, 0.001, 0.0005]
spine_bonus = 0

for idx,TFA in enumerate(np.concatenate((np.arange(0.01, 0.001, -0.001), np.arange(0.001, 0.0004, -0.0001)))):

    S = len(cut_srlgs)
    H = np.array([Hnull+spine_bonus*g.edges[e]['onspine'] for e in g.edges])

    print(f'The shape of the intensity matrix: {intensity.shape}')
    print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

    start = time.perf_counter()

    #Model
    model = Model(sense=MINIMIZE, solver_name=GRB)

    #Variables
    deltaH = [model.add_var(var_type=INTEGER, lb=0, ub=4) for l,_ in enumerate(g.edges)]
    Z = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in range(S)]
    Y = [[[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters] for i in links]
    W = [[model.add_var(var_type=BINARY) for k in magnitudes] for j in epicenters]
    print("%.1f s:\tVáltozók létrehozva..."%(time.perf_counter()-start))
    print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')


    #Objective Function
    model.objective = xsum( g.edges[link_id]['length'] * deltaH[link_idx] for link_idx,link_id in enumerate(g.edges) )
    print("%.1f s:\tCélfüggvény létrehozva..."%(time.perf_counter()-start))
    print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')


    #Constraint 1
    for l,p,m in product(*[links, epicenters, magnitudes]):
        model.add_constr( Y[l][p][m] >= 1 - ((H[l] + deltaH[l]) / intensity[l,p,m]) )
    print("%.1f s:\tElső egyenlet létrehozva..."%(time.perf_counter()-start))
    print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')


    #Constraint 2
    for (c,s), p, m in product(*[enumerate(cut_srlgs), epicenters, magnitudes]):
        model.add_constr( Z[c][p][m] >= (xsum(Y[list(g.edges).index(linkID)][p][m] for linkID in s) - len(s) + 1) )
    print("%.1f s:\tMásodik egyenlet létrehozva..."%(time.perf_counter()-start))
    print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

    #Constraint 3
    for (c,s), p, m in product(*[enumerate(cut_srlgs), epicenters, magnitudes]):
        model.add_constr( W[p][m] >= Z[c][p][m] )

    model.add_constr(xsum( W[p][m] * prob_matrix[p,m] for p,m in product(epicenters,magnitudes) ) <= TFA )
    print("%.1f s:\tHarmadik egyenlet létrehozva..."%(time.perf_counter()-start))
    print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

    #Start optimization
    model.optimize()#max_seconds=
    runtime_ILP = time.perf_counter()-start
    print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

    # The result
    cost = 0
    selected = []
    dH = []
    for l,e in enumerate(g.edges):
        dH.append(int(deltaH[l].x + 0.1))
        if deltaH[l].x >= 0.5:
            selected.append((deltaH[l].x,e, g.edges[e]['length']))
            cost += deltaH[l].x * g.edges[e]['length']
    print(cost)
    print(*selected, sep='\n')
    H = H + np.array(dH)

    for T in Ts:
        active_srlgs = [srlg for srlg in all_srlgs if get_SRLG_probability(srlg, g, intensity, H, prob_matrix)>T]
        write_networkx_to_srg(f'results/{network_name}/{network_name}_TFA{TFA:.4f}_T{T}_H2_SB{spine_bonus}.srg', g, active_srlgs)

    # Saving the result
    df_cost = pd.read_csv(f'results/{network_name}/comparison_{network_name}_SB{spine_bonus}.csv')
    df_cost.loc[idx,'Runtime ILP'] = runtime_ILP
    df_cost.loc[idx,'Cost ILP'] = cost
    df_cost.to_csv(f'results/{network_name}/comparison_{network_name}_SB{spine_bonus}.csv', index=False, float_format='%.4f')
    
    df_H = pd.read_csv(f'results/{network_name}/upgrade_{network_name}_TFA{TFA:.4f}_SB{spine_bonus}.csv')
    df_H['Delta H (ILP)'] = dH
    df_H['H (ILP)'] = H
    dfH.to_csv(f'results/{network_name}/upgrade_{network_name}_TFA{TFA:.4f}_SB{spine_bonus}.csv', index=False, float_format='%.4f')
