import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from geopy.distance import distance, lonlat
from tqdm.notebook import tqdm
import xml.etree.ElementTree as ET
from ast import literal_eval
import re
import pickle


# Creating the DataFrame of the earthquakes' probabilities
df = pd.read_table('../incremental_annual_rates/incremental_annual_rates_italy.txt', sep='\s+', header=None)

df.columns = ['Lat', 'Long'] + [str(np.around(x,1)) for x in np.arange(4.6, 8.7, 0.1)]
df.drop(df.columns[42:], axis=1, inplace=True)
df.fillna(value=0.0, inplace=True)
df = df[df['4.6'] get_ipython().getoutput("= 0.0]")
df.rename(columns={'Lat':'Long', 'Long':'Lat'}, inplace=True)

df.to_csv('incremental_annual_rates/incremental_annual_rates_italy_without_zeros.csv', index=False)


df = pd.read_csv('../incremental_annual_rates/incremental_annual_rates_italy.csv')
#df.set_index(['Lat','Long'], inplace=True)
df.head(2)


df['Lat'] = (df['Lat']*5).astype(int)
df['Long'] = (df['Long']*5).astype(int)
df_4 = df.groupby(['Lat','Long']).sum()
df_4 = df_4[df_4['4.6'] get_ipython().getoutput("= 0.0]")
df_4.reset_index(inplace=True)
df_4['Lat'] = df_4['Lat']/5
df_4['Long'] = df_4['Long']/5
df_4


df_4.to_csv('earthquake_probabilities/italy_ds16.csv', index=False)


g = nx.read_gml('networks/italy.gml', label='id')


def intensity_europe(M, R):
    h = 3.91
    D = np.sqrt(h**2 + R**2)
    return 1.621*M - 1.343 - 0.0086*(D-h) - 1.037*(np.log(D)-np.log(h))


l = [tuple(d.values()) for d in g.edges[('Rome', 'Pescara', 0)]['points']['point']]
edge = LineString(l)
point = Point(10.8,48.3)
nearest_point = nearest_points(edge, point)[0]
nearest_point.coords[:]


nx.write_gml(g, 'networks/italy_withLength.gml')


for edge in g.edges:
    e = LineString([tuple(d.values()) for d in g.edges[edge]['points']['point']])
    p = list(e.coords)
    l = 0
    for x,y in zip(p[:-1], p[1:]):
        l += distance(lonlat(*x), lonlat(*y)).km
    g.edges[edge]['length'] = l
    print(edge, l)


for e in g.edges:
    edge_points = [tuple(p.values()) for p in g.edges[e]['points']['point']]
    edge = LineString(edge_points)
    epicenter = Point(10.5,48.2)
    nearest_point = nearest_points(edge, epicenter)[0].coords[0]
    print(nearest_point)
    print(epicenter.coords[0])
    dist = distance(lonlat(*epicenter.coords[0]), lonlat(*nearest_point)).km
    intensity = intensity_europe(8.6, dist)
    print("get_ipython().run_line_magic(".1f"", " % intensity)")


#df['Epicenter'] = list(zip(df.Lat, df.Long))
df.insert(loc=0, column='Epicenter', value=list(zip(df.Lat, df.Long)))
df.set_index('Epicenter', inplace=True)
df.drop(['Lat', 'Long'], axis=1, inplace=True)


#df=df_4
df.head()


epicenters = df.index.to_numpy()
magnitudes = df.columns.to_numpy()


I = np.ones((g.number_of_edges(), len(epicenters), len(magnitudes)))
for i, e in tqdm(enumerate(g.edges), total=g.number_of_edges()):
    edge_points = [tuple(p.values()) for p in g.edges[e]['points']['point']]
    l = LineString(edge_points)
    for j, epicenter in tqdm(enumerate(epicenters), total=len(epicenters)):
        p = Point(epicenter)
        nearest_point = nearest_points(l, p)[0].coords[0]
        R = distance(lonlat(*epicenter), lonlat(*nearest_point)).km
        for k, M in enumerate(magnitudes):
            intensity = intensity_europe(np.float(M), R)
            if intensity > 1.0:
                I[i,j,k] = intensity


np.save('intensities/italy_ds8.npy', I)


def remains_connected(g, srlg):
    g.remove_edges_from(srlg)
    return nx.is_connected(g.to_undirected())

def get_minimal_cut_SRLGs(PSRLG_file):
    xtree = ET.parse(PSRLG_file)
    xroot = xtree.getroot()
    cut_srlgs = []
    for fs in tqdm(xroot.iter('Failure_State'), total=len(xroot.findall('Failure_State'))):
        # (edge_id, node_id, node_id)
        srlg = set([ (int(edge[1]), int(edge[2]), 0) if edge[0] get_ipython().getoutput("= '24' else (int(edge[1]), int(edge[2]), 1) for edge in re.findall(r"(\d+):\((\d+)\D+(\d+)\D+\)", fs.find('Edges').text)])")
        probability = float(fs.find('Probability').text.strip())
        if not remains_connected(g.copy(), srlg) and probability>10**-4:
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

min_cut_srlgs = get_minimal_cut_SRLGs('PSRLGs/whole_graph_complete_VI_upper_big_grid.xml')


# Correcting node order in SRLG edges
edge_list = list(g.edges)
for idx,srlg in enumerate(min_cut_srlgs):
    for l in srlg:
        if l not in edge_list:
            min_cut_srlgs[idx].discard(l)
            min_cut_srlgs[idx].add((l[1],l[0],l[2]))


# Write
with open('min_cut_SRLGs/italy_complete_10-4', 'wb') as fp:
    pickle.dump(min_cut_srlgs, fp)

# Read
#with open ('min_cut_SRLGs/italy', 'rb') as fp:
#    cut_srlgs = pickle.load(fp)


min_cut_srlgs
