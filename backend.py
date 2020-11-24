import pandas as pd
import numpy as np
import networkx as nx
from geopy.distance import distance, lonlat
import xml.etree.ElementTree as ET
from ast import literal_eval
import re
import pickle
from svector import SVector
from tqdm.notebook import tqdm

# Intensity calculation
def intensity_europe(M, R):
    h = 3.91
    D = np.sqrt(h**2 + R**2)
    return 1.621*M - 1.343 - 0.0086*(D-h) - 1.037*(np.log(D)-np.log(h))

def intensity_usa(M, R):
    h = 10.
    D = np.sqrt(h**2 + R**2)
    return 0.44 + 1.70*M - 0.0048*D - 2.73*np.log10(D)


# Distance calculation with Svector class
def node_to_SVector(node):
    lat, long = node['Latitude'], node['Longitude']
    return SVector(lat, long)

def edge_to_SVectors(edge):
    lat0, long0 = edge['points']['point'][0]['Latitude'], edge['points']['point'][0]['Longitude']
    lat1, long1 = edge['points']['point'][1]['Latitude'], edge['points']['point'][1]['Longitude']
    return SVector(lat0, long0), SVector(lat1, long1)


# Graph and SRLG calculations
def remains_connected(g, srlg):
    g.remove_edges_from(srlg)
    return nx.is_connected(g.to_undirected())

def reamins_k_connected(g, srlg):
    g.remove_edges_from(srlg)
    return nx.is_k_edge_connected(g.to_undirected())

def get_minimal_cut_SRLGs(PSRLG_file, g):
    xtree = ET.parse(PSRLG_file)
    xroot = xtree.getroot().find('PSRLGList')
    cut_srlgs = []
    for fs in tqdm(xroot.iter('PSRLG'), total=len(xroot.findall('PSRLG'))):
    #for fs in tqdm(xroot.iter('Failure_State'), total=len(xroot.findall('Failure_State'))):
        # (edge_id, node_id, node_id)
        #srlg = set([ (int(edge[1]), int(edge[2]), 0) if edge[0] != '24' else (int(edge[1]), int(edge[2]), 1) for edge in re.findall(r"(\d+):\((\d+)\D+(\d+)\D+\)", fs.find('Edges').text)])
        srlg = set([ (int(edge[1]), int(edge[2]), 0) for edge in re.findall(r"(\d+):\((\d+)\D+(\d+)\)", fs.find('Edges').text)])
        probability = float(fs.find('Probability').text.strip())
        #print(fs.find('Edges').text)
        #print(probability, srlg)
        if not remains_connected(g.copy(), srlg):# and probability>10**-5:
            added = False
            for i, s in enumerate(cut_srlgs):
                if srlg.issubset(s):
                    cut_srlgs[i] = srlg
                    added = True
                    break
                elif srlg.issuperset(s):
                    added = True
                    break
            if not cut_srlgs or not added:
                cut_srlgs.append(srlg)
    return cut_srlgs

def get_SRLGs(PSRLG_file):
    xtree = ET.parse(PSRLG_file)
    xroot = xtree.getroot().find('PSRLGList')
    srlgs = []
    probabilities = []
    rates = []
    for fs in xroot.iter('PSRLG'):
    #for fs in tqdm(xroot.iter('Failure_State'), total=len(xroot.findall('Failure_State'))):
        # (edge_id, node_id, node_id)
        #srlg = set([ (int(edge[1]), int(edge[2]), 0) if edge[0] != '24' else (int(edge[1]), int(edge[2]), 1) for edge in re.findall(r"(\d+):\((\d+)\D+(\d+)\D+\)", fs.find('Edges').text)])
        srlg = set([ (int(edge[1]), int(edge[2]), 0) for edge in re.findall(r"(\d+):\((\d+)\D+(\d+)\)", fs.find('Edges').text)])
        probability = float(fs.find('Probability').text.strip())
        #rate = float(fs.find('Rate').text.strip())
        #print(probability, srlg)
        #if probability > 10**-5:
        if srlg:
            srlgs.append(srlg)
            probabilities.append(probability)
            #rates.append(rate)
    return srlgs, probabilities#, rates


def get_SRLG_probability(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix):
    edges = list(network.edges)
    srlg_occur = np.full(intensity_matrix.shape[1:], fill_value=True, dtype=bool)
    for l in srlg:
        l_idx = edges.index(l)
        srlg_occur &= intensity_matrix[l_idx] > intensity_tolerance[l_idx]
    return probability_matrix[srlg_occur].sum()


# Networkx and lgf conversions
def read_lgf_to_networkx_extended(lgf_file):
    file = open(lgf_file, 'r')
    all_line = file.read().split('\n')
    # edge: u v label [length] onspine unavInit unavFinal
    all_match = [re.findall(r'^(\d+)\t\((.+),(.+)\)|^(\d+)\t(\d+)\t(\d+)\t*(\S{0})\t*(\d+)\t*(\S+)\t*(\S*)|(^\d+-\d+( \d+-\d+)+)', line) for line in all_line]
    all_match = filter(len, all_match)
    G = nx.MultiGraph()
    SRLGs = []
    for line in all_match:
        line = line[0]
        if line[0]:
            #G.add_node(int(line[0]), pos=(float(line[1]),float(line[2])))
            G.add_node(int(line[0]), Longitude=float(line[1]), Latitude=float(line[2]))
        elif line[3]:
            u, v = int(line[3]), int(line[4])
            edge_key = G.add_edge(u,v)
            G.edges[u, v, edge_key]['points'] = {'point': [
                {'Longitude': G.nodes[u]['Longitude'], 'Latitude': G.nodes[u]['Latitude']},
                {'Longitude': G.nodes[v]['Longitude'], 'Latitude': G.nodes[v]['Latitude']}]}
            if line[6]:
                length = float(line[6])
                G.edges[u, v, edge_key]['length'] = length
            if line[9]:
                onspine = int(line[7])
                unav_1 = float(line[8])
                unav_k = float(line[9])
                G.edges[u, v, edge_key]['onspine'] = onspine
                G.edges[u, v, edge_key]['unav1'] = unav_1
                G.edges[u, v, edge_key]['unav'] = unav_k
            elif line[8]:
                onspine = int(line[7])
                unav = float(line[8])
                G.edges[u, v, edge_key]['onspine'] = onspine
                G.edges[u, v, edge_key]['unav1'] = unav
                G.edges[u, v, edge_key]['unav'] = unav
        if line[10]:
            srlg = line[10]
            SRLGs.append([ tuple(map(int, link.split('-'))) for link in srlg.split()])
    if SRLGs:
        return G, SRLGs
    else:
        return G


def write_networkx_to_lgf(G, network_name, extended=False):
    f = open(str(network_name), 'w')
    f.write('@nodes\n')
    f.write('label\tcoords\n')
    for node, attr in G.nodes(data=True):
        f.write(str(node) + '\t(' + str(attr['Longitude']) + ',' + str(attr['Latitude']) + ')\n')
    f.write('@edges\n')
    if extended:
        f.write('\t\tlabel\tonspine\tunav_1\tunav\n')
    else:
        f.write('\t\tlabel\tonspine\tunav\n')
    for label,(u,v,k) in enumerate(G.edges):
        e = G[u][v][k]
        if extended:
            line = str("%d\t%d\t%d\t%d\t%.10f\t%.6f\n" % (u, v, label, e['onspine'], e['unav_1'], e['unav']))
        else:
            line = str("%d\t%d\t%d\t%d\t%.10f\n" % (u, v, label, e['onspine'], e['unav']))
        f.write(line)
    f.close()
    return 0


def write_networkx_to_srg(network_name, G, SRLGs):
    f = open(str(network_name), 'w')
    f.write('@nodes\n')
    f.write('label\tcoords\tunav\n')
    for node, attr in G.nodes(data=True):
        f.write(str(node) + '\t(' + str(attr['Latitude']) + ',' + str(attr['Longitude']) + ')\t0\n')
    f.write('@edges\n')
    f.write('\t\tlabel\tonspine\tunav\n')
    for label,(u,v) in enumerate(G.edges()):
        e = G[u][v][0]
        line = str("%d\t%d\t%d\t%d\t%.10f\n" % (u, v, label, e['onspine'], e['unav']))
        f.write(line)
    for label,(u,v) in enumerate(reversed(list(G.edges())), start=len(G.edges)):
        e = G[u][v][0]
        line = str("%d\t%d\t%d\t%d\t%.10f\n" % (v, u, label, e['onspine'], e['unav']))
        f.write(line)
    f.write('@srgs\n')
    for i, srlg in enumerate(SRLGs):
        f.write(f'{str(i+1)} 0\n')
        line = ''
        for e in srlg:
            line += f'{e[0]}-{e[1]} {e[1]}-{e[0]} '
        f.write(line + '\n')
    f.close()
    return 0


# Heuristics

def get_probability_of_falling_apart(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix):
    edges = list(network.edges)
    cut_occur = np.full(intensity_matrix.shape[1:], fill_value=False, dtype=bool)
    for srlg in srlgs:
        srlg_occur = np.full(intensity_matrix.shape[1:], fill_value=True, dtype=bool)
        for l in srlg:
            l_idx = edges.index(l)
            srlg_occur &= intensity_matrix[l_idx] > intensity_tolerance[l_idx]
        cut_occur |= srlg_occur
    return probability_matrix[cut_occur].sum()

def countSRLGlinks(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix):
    edges = list(network.edges)
    partofSRLG = np.zeros(len(edges), dtype=float)
    for srlg in srlgs:
        srlg_probability = get_SRLG_probability(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix)
        for l in srlg:
            partofSRLG[edges.index(l)] += srlg_probability
    #return dict(zip(edges, partofSRLG))
    return partofSRLG

def get_edge_to_improve_1(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix):
    edges = list(network.edges)
    partofSRLG = countSRLGlinks(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix)
    partofSRLG = partofSRLG * (intensity_tolerance < 8.5)
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

def get_edge_to_improve_2(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
    edges = list(network.edges)
    probability_reduction_values = np.zeros(len(edges))
    probability_of_falling_apart = get_probability_of_falling_apart(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix)
    
    for idx, edge in enumerate(edges):
        probability_reduction = 0
        if intensity_tolerance[idx] < 8.5:
            intensity_tolerance[idx] += 1
            decreased_probability_of_falling_apart = get_probability_of_falling_apart(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix)
            probability_reduction = probability_of_falling_apart - max(threshold, decreased_probability_of_falling_apart)
            intensity_tolerance[idx] -= 1
        probability_reduction_values[idx] = probability_reduction / network.edges[edge]['length']
    
    return np.argmax(probability_reduction_values)

def heuristic(version, srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
    active_srlgs = srlgs.copy()
    edges = list(network.edges)
    cost = 0
    
    probability_of_falling_apart = get_probability_of_falling_apart(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix)
    #print(f'{probability_of_falling_apart:.5f}')
    
    while probability_of_falling_apart > threshold:
        if version == 1:
            edge_to_improve = get_edge_to_improve_1(active_srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix)
        else:
            edge_to_improve = get_edge_to_improve_2(active_srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
        cost += network.edges[edges[edge_to_improve]]['length']
        #print(edge_to_improve)
        intensity_tolerance[edge_to_improve] += 1
        probability_of_falling_apart = get_probability_of_falling_apart(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix)
        #print(f'{probability_of_falling_apart:.5f}')
        #print(intensity_tolerance)
        
    print(f'H{version} Cost: {cost:.0f}')
    return intensity_tolerance, cost




# # Old Heuristic 1 & 2

def get_SRLG_probability_matrix(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix):
    edges = list(network.edges)
    srlg_occur = np.full(intensity_matrix.shape[1:], fill_value=True, dtype=bool)
    for l in srlg:
        l_idx = edges.index(l)
        srlg_occur &= intensity_matrix[l_idx] > intensity_tolerance[l_idx]
    return probability_matrix[srlg_occur]

# def remove_improbable_SRLGs(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
#     active_srlgs = srlgs.copy()
#     for s in active_srlgs.copy():
#         p = get_SRLG_probability_matrix(s, g, intensity, H, prob_matrix).sum()
#         if p < threshold:
#             active_srlgs.remove(s)
#     return active_srlgs

# def countSRLGlinks(srlgs, network):
#     edges = list(network.edges)
#     partofSRLG = np.zeros(len(edges), dtype=int)
#     for srlg in srlgs:
#         for l in srlg:
#             partofSRLG[edges.index(l)] += 1
#     #return dict(zip(edges, partofSRLG))
#     return partofSRLG

# def get_edge_to_improve_1(srlgs, network):
#     edges = list(network.edges)
#     partofSRLG = countSRLGlinks(srlgs, network)
#     max_indexes = [i for i, j in enumerate(partofSRLG) if j == max(partofSRLG)]
#     #print(max_indexes)
#     if len(max_indexes) > 1:
#         max_index = max_indexes[0]
#         min_length = network.edges[edges[max_index]]['length']
#         for idx in max_indexes:
#             length = network.edges[edges[idx]]['length']
#             if length < min_length:
#                 max_index = idx
#                 min_length = length
#         return max_index
#     else:
#         return max_indexes[0]

# def heuristic_1(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
#     active_srlgs = remove_improbable_SRLGs(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
#     cost = 0
#     edges = list(network.edges)
#     print(len(active_srlgs))
#     while len(active_srlgs):
#         edge_to_improve = get_edge_to_improve_1(active_srlgs, network)
#         cost += network.edges[edges[edge_to_improve]]['length']
#         #print(edge_to_improve)
#         intensity_tolerance[edge_to_improve] += 1
#         active_srlgs = remove_improbable_SRLGs(active_srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
#     print(f'H1 Cost: {cost:.0f}')
#     return intensity_tolerance, cost

# def get_SRLG_probability_reduction(link_idx, srlg, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
#     srlg_prob_matrix = get_SRLG_probability_matrix(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix)
#     srlg_prob = srlg_prob_matrix.sum()
#     intensity_tolerance[link_idx] += 1
#     new_srlg_prob_matrix= get_SRLG_probability_matrix(srlg, network, intensity_matrix, intensity_tolerance, probability_matrix)
#     new_srlg_prob = new_srlg_prob_matrix.sum()
#     intensity_tolerance[link_idx] -= 1
#     return srlg_prob - new_srlg_prob

# def get_edge_to_improve_2(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
#     edges = list(network.edges)
#     probability_reduction_values = np.zeros(len(edges))
#     for idx, edge in enumerate(edges):
#         probability_reduction = 0
#         for srlg in srlgs:
#             if edge in srlg:
#                 probability_reduction += get_SRLG_probability_reduction(idx, srlg, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
#         probability_reduction_values[idx] = probability_reduction / network.edges[edge]['length']
#     return np.argmax(probability_reduction_values)

# def heuristic_2(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold):
#     active_srlgs = remove_improbable_SRLGs(srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
#     cost = 0
#     edges = list(network.edges)
#     print(len(active_srlgs))
#     while len(active_srlgs):
#         edge_to_improve = get_edge_to_improve_2(active_srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
#         cost += network.edges[edges[edge_to_improve]]['length']
#         #print(edge_to_improve)
#         intensity_tolerance[edge_to_improve] += 1
#         active_srlgs = remove_improbable_SRLGs(active_srlgs, network, intensity_matrix, intensity_tolerance, probability_matrix, threshold)
#     print(f'H2 Cost: {cost:.0f}')
#     return intensity_tolerance, cost