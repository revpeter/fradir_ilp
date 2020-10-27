from mip import *
from backend import *
from itertools import product
import time
import resource

network_name = 'italy_995'
spine_bonus = 1

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
Ts = [0.01, 0.005, 0.001, 0.0005]

#for idx, TFA in enumerate([0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005]):
for idx, TFA in enumerate([0.0009, 0.0008, 0.0007, 0.0006, 0.0005], 10):

    start = time.perf_counter()
    H = np.array([Hnull+spine_bonus*g.edges[e]['onspine'] for e in g.edges])

    #Load model
    model = Model(sense=MINIMIZE, solver_name=GRB)
    model.read(f'results/{network_name}/{network_name}_SB{spine_bonus}.lp')
    model.constrs[-1].rhs = TFA

    #Start optimization
    model.optimize()#max_seconds=
    runtime_ILP = time.perf_counter()-start
    print(f'Memory usage: {int(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000)} MB')

    # The result
    deltaH = model.vars[:L]
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

    if TFA in Ts:
        for T in Ts:
            active_srlgs = [srlg for srlg in all_srlgs if get_SRLG_probability(srlg, g, intensity, H, prob_matrix)>T]
            write_networkx_to_srg(f'results/{network_name}/SRLG/SB{spine_bonus}/{network_name}_TFA{TFA:.4f}_T{T}_ILP_SB{spine_bonus}.srg', g, active_srlgs)

    # Saving the result
    df_cost = pd.read_csv(f'results/{network_name}/comparison_{network_name}_SB{spine_bonus}.csv')
    df_cost.loc[idx,'Runtime ILP'] = runtime_ILP
    df_cost.loc[idx,'Cost ILP'] = cost
    df_cost.to_csv(f'results/{network_name}/comparison_{network_name}_SB{spine_bonus}.csv', index=False, float_format='%.4f')
    
    df_H = pd.read_csv(f'results/{network_name}/upgrade_level/SB{spine_bonus}/upgrade_{network_name}_TFA{TFA:.4f}_SB{spine_bonus}.csv')
    df_H['Delta H (ILP)'] = dH
    df_H['H (ILP)'] = H
    df_H.to_csv(f'results/{network_name}/upgrade_level/SB{spine_bonus}/upgrade_{network_name}_TFA{TFA:.4f}_SB{spine_bonus}.csv', index=False, float_format='%.4f')

