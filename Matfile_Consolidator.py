'''
Author: Leila Thompsky (based on code by Grace Li)
Date: 1/10/2025

This script calculates the results from our simulations of our adaptive edge-weighted DW model. Run this script after
running simulations using either synthetic_networks_experiment.py or netscience_experiment.py

After running this script, the linePlots scripts can be used to generate plots of our simulation results.

To run this script, at the top, modifiy the variables that give the desired graph, number of nodes, random graph 
parameters (if applicable), and model parameters (the edge-weight increase parameter gamma, the edge-weight decrease parameter delta, 
the initial edge-weight z-- this is called z0 in the paper--, and the confidence bound c).

For each simulation, the
    - random-graph parameters (if applicable)
    - graph number (for random graphs)
    - edge-weight increase parameter gamma
    - edge-weight decrease parameter delta
    - confidence bound c
    - opinion set (abbreviated to op in the code), 
are used to load the stored matfile for that simulation.

The following quantities are calculated at convergence time for each simulation:
    - number of opinion clusters
    - number of major opinion clusters
    - number of minor opinion clusters
    - maximum pairwise difference in opinion of nodes in the same opinion cluster (we call this the opinion diameter of the cluster)
    - convergence time (as a number of time steps)
    - whether or not the bailout time is reached (as a boolean)
    - mean local receptiveness
    - Shannon entropy H
    - mean, variance, skew and kurtosis of opinions
We store these calculated results as a row in a csv file. For each (gamma, delta) pair, we generate and save a csv file
of simulation results and store it in the "combined_results" folder for that graph. 

IMPORTANT: This script requires igraph version 0.9.7 as the syntax of that version is used to calculate our various quantities.

'''

import numpy as np
import pandas as pd
from scipy import io, stats
import os
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import time as time
import igraph as igraph
import sys

print("igraph version:", igraph.__version__)


#percent threshold for minor opinion clusters. 
#if the number of nodes in a cluster is <= minor_clusters_percent, we count it as a minor cluster
minor_clusters_percent = 1

## Change parameters here ---------------------------------------------------------------
facebook_networks = ["Caltech", "Pepperdine", "Reed", "Rice", "Swarthmore", "UCSB"]
graph_type = "complete"
ns = [200] #have set in edge_weighted_experiment 
graph_name = "netscience" #If we have a specific graph 


network_dataset = False
if network_dataset:
    graph_type = graph_name
    ns = [1]

#Edge probability for ER graphs
ps = [1] #by default there are no edge probabilities, we put a single value as placeholder
if graph_type == "erdos-renyi":
    ps = [0.1] #edge probability for ER graph


#value of k for degree-regular cycle-like graphs (for convenience, we refer to these as 'degree-regular graphs' throughout the code) 
if graph_type == "degree-regular":
    k = 300

runtoTmax = False
toltest = False
tol = .01 #specify the tolerance value if its not the standard

max_times_continued = 1 #Maximum times we continued the simulation so we check each folder



#Set model parameters
gammas = [0.1,0.5,0.9] #edge-weight increase parameters
deltas = [0.1, 0.5, 0.9] #edge-weight decrease parameters



delta_gammas = [(1, 0)] 

for delta in deltas:
    for gamma in gammas:
        delta_gammas.append((delta, gamma))

#Confidence bound
cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
#Intial edge weights
zs = [0.1, 0.5, 0.9] 
#Compromise parameter
mus = [0.1, 0.3, 0.5] 

#specify opinions sets
opinion_sets = list(range(20))

#Specify which graphs to look at if using a random-graph model
graphs = list(range(0,5))



## Some initial set up ---------------------------------------------------------------
#Get number of graphs and opinion sets
n_graphs = len(graphs)
n_opinion_sets = len(opinion_sets)

#savefolder name for experiment
folder_name_dict = {"complete": "Complete", "erdos-renyi": "Erdos-Renyi", 'degree-regular': 'Degree-Regular'}
random_graph_folders = ["Erdos-Renyi"] #The names of the folders with random-graph models
    
def generate_row(results, minor_threshold, graph):
    '''
    Given the resulting output from opening a stored matfile, put together the row entry
    of the Pandas DataFrame to store the results
    
    minor_threhold: int,
        For cluster sizes <= minor threshold, we consider them minor clusters, otherwise we consider them major clusters
    graph: igraph graph
    '''
    
    #Calculate the average local receptiveness
    avg_local_receptiveness = np.mean(results['local_receptiveness'][0])
        
    #Calculate the Shannon entropy for the normalized clusters
    n_clusters = results['n_clusters'][0][0]
    n_minor = 0 #start counting the number of minor clusters
    #If we reached the bailout time, then there's no clusters found and we set everything to NaN
    if n_clusters == 0:
        n_clusters = np.nan
        n_minor = np.nan
        n_major = np.nan
        n_clusters_major = np.nan
        H = np.nan
        max_diameter = np.nan
    #If we sucessfully determined the clusters, then we calculate the Shannon entropy and max cluster diameter
    else:
        diameters = []
        cluster_sizes = [] #get and store the size of each opinion cluster
        for i in range(n_clusters):
            key = "cluster" + str(i)
            cluster = results[key][0]
            cluster_sizes.append(len(cluster))
            if len(cluster) <= minor_threshold:
                n_minor += 1
            final_ops = results['final_opinions'][0][cluster]
            diameters.append(max(final_ops) - min(final_ops)) 
        cluster_sizes = np.array(cluster_sizes) / np.sum(np.array(cluster_sizes)) #normalize the cluster sizes
        #Calculate Shannon's entropy of normalized clusters
        H = - np.sum(cluster_sizes * np.log(cluster_sizes))
        
        n_major = n_clusters - n_minor
    
        max_diameter = max(diameters)
        
    #Calculate the mean, variance, skewness, and kurtosis of the final opinion vector
    final_opinions = results['final_opinions'][0]
    op_mean = np.mean(final_opinions)
    op_var = np.var(final_opinions)
    op_skew = stats.skew(final_opinions, axis=None)
    op_kurtosis = stats.kurtosis(final_opinions, axis=None, fisher=True)
    

    #Reconstruct the effective graph at T and look at the edges remaining
    G = graph.copy()
    n = G.vcount()
    edge_start = np.array([e.tuple[0] for e in G.es], dtype=np.int64)
    edge_end = np.array([e.tuple[1] for e in G.es], dtype=np.int64)
    
    G.vs['opinion'] = final_opinions
    opinion_diff = np.array(G.vs[edge_start.tolist()]['opinion']) - np.array(G.vs[edge_end.tolist()]['opinion'])
    opinion_diff = np.abs(opinion_diff)
    
    columns = ['delta', 'gamma', 'c', 'z', 'mu', 'opinion_set',
               'n_clusters', 'n_major', 'n_minor',
               'max_diameter', 'T', 'bailout', 
               'avg_opinion_diff', 'avg_local_receptiveness', 'entropy',
               'op_mean', 'op_var', 'op_skew', 'op_kurtosis',
              ]
    
    row = pd.DataFrame(columns = columns)
    row.loc[0] = [delta, gamma, c, z, mu, opinion_set,
                  n_clusters, n_major, n_minor,
                  max_diameter,
                  results['T'][0][0],
                  results['bailout'][0][0],
                  results['avg_opinion_diff'][0][0],
                  avg_local_receptiveness, H,
                  op_mean, op_var, op_skew, op_kurtosis,
                 ]
    
    return row

# ---------------------------------------------------------------
'''
Read the stored matfiles and create a pandas dataframe of simulation results for each run
including number of clusters, convergence time, number of positive interactions (T_changed), 
and average opinion distance
'''
counter = 0    
for n in ns:
    
    print('n =', n)
    
    if network_dataset:
        if graph_name in facebook_networks:
            experiment = 'college-networks/' + graph_name + ("/runtoTmax" * runtoTmax)
        elif graph_name == "netscience":
            experiment = "netscience" + ("/runtoTmax" * runtoTmax)
        folder_name = graph_name
        get_n = True
        
    else:
        folder_name = folder_name_dict[graph_type]
        # experiment = folder_name + '/' + graph_type + str(n) + ("/runtoTmax" * runtoTmax)
        experiment = graph_type + str(n) + ("/runtoTmax" * runtoTmax)
        minor_threshold = math.floor(n * minor_clusters_percent/100)  
        print('\nn = ', n)
        print('The minor threshold is: ', minor_threshold)
        get_n = False

    if folder_name not in random_graph_folders:
        graphs = [1] #we don't have multiple graphs to consider if it's not a random-graph model
    
    for p in ps:  
        txtfile = "/MatfileConsolidatorOutputs"
        if graph_type == "erdos-renyi":
            txtfile = txtfile + f"--p{p}"
        elif graph_type == "SBM":
            txtfile = txtfile + f"--p_aa{p_aa}-p_bb{p_bb}-p_ab{p_ab}"
        elif graph_type == "degree-regular": 
            txtfile = txtfile + f"--k{k}"
        txtfile = txtfile + ".txt"

        with open(experiment + txtfile, 'w') as f:

            if graph_type == "erdos-renyi":
                print('---------- ER with p =', p, '-------------', file=f, flush=True)
            elif graph_type == "SBM":
                print(f"---------- SBM with p_aa = {p_aa}, p_bb = {p_bb}, p_ab = {p_ab} -------------", file=f, flush=True)
                
            if not network_dataset:
                print('\nn = ', n, file=f, flush=True)
                print('The minor threshold is: ', minor_threshold, file=f, flush=True)

            #Read the stored matfiles and create a pandas dataframe of simulation results for each run
            #including number of clusters, convergence time, effective convergence time, and average opinion distance

            #Initialize dataframe
            #Specify column names for the storage dataframe
            
            columns = ['delta', 'gamma', 'c', 'z', 'mu','opinion_set',
                       'n_clusters', 'n_major', 'n_minor',
                       'max_diameter', 'T', 'bailout', 
                       'avg_opinion_diff', 'avg_local_receptiveness', 'entropy',
                       'op_mean', 'op_var', 'op_skew', 'op_kurtosis',
                      ]

            if folder_name in random_graph_folders:
                columns = ['graph'] + columns

            #Fill in the data frame row by row and store
            #We will create a new dataframe for each delta, gamma pair
            for pair in delta_gammas:
                delta = pair[0]
                gamma = pair[1]

                print('\ndelta =', delta, ' and gamma = ', gamma, file=f, flush=True)
                print('\ndelta =', delta, ' and gamma = ', gamma)

                total_bailout = 0 #keep track of how many times we hit the bailout time
                file_not_found = 0 #keep a track of the count of how many files are not done yet
                
                # Get the random graph seed if its a random graph
                if folder_name in random_graph_folders:
                    graph_seed_file = f"{graph_type}{n}/graph_seeds.csv"
                    print(graph_seed_file)
                    df = pd.read_csv(graph_seed_file)
                    if graph_type == 'erdos-renyi':
                        df = df[df['p'] == p]
                    elif graph_type == "SBM":
                        df = df[df['p_aa'] == pref_matrix[0][0]]
                        df = df[df['p_ab'] == pref_matrix[0][1]]
                        df = df[df['p_bb'] == pref_matrix[1][1]]  
                    graph_seed = df['graph_seed'].values[0]
                    graph_seed = int(graph_seed)
                    print(graph_seed)

                #Initialize the dataframe
                df = pd.DataFrame(columns = columns)
                    
                #Fill in the data frame row by row and store for each graph number
                for graph_number in graphs:
                    
                    graph_to_adjacency = False

                    if graph_type == "complete":
                        G = igraph.Graph.Full(n)

                    elif folder_name in random_graph_folders:
                        print('Graph', graph_number)
                        print('Graph', graph_number, file=f, flush=True)
                        #Reinitialize the random seed and generate the corresponding graph number from that seed
                        if graph_type == "erdos-renyi":
                            random_graph = np.random.default_rng(graph_seed)
                            for i in range(graph_number + 1):
                                seed = random_graph.integers(low=0, high=sys.maxsize)
                                random.seed(a=seed)
                                G = igraph.Graph.Erdos_Renyi(n, p)
                        elif graph_type == "SBM":
                            random_graph = np.random.default_rng(graph_seed)
                            #Block sizes
                            A = int(n * 0.75)
                            B = int(n * 0.25)
                            block_sizes = [A, B]
                            for i in range(graph_number + 1):
                                seed = random_graph.integers(low=0, high=sys.maxsize)
                                random.seed(a=seed)
                                G = igraph.Graph.SBM(n, pref_matrix, block_sizes, directed=False, loops=False)
                                
                        if graph_to_adjacency:
                            graph_file = folder_name_dict[graph_type] + f"/{graph_type}{n}/adjancency_matrix/"
                            if not os.path.exists(graph_file):
                                os.makedirs(graph_file)
                            graph_file = graph_file + f"graph{graph_number}.csv"
                            G.write_adjacency(graph_file, sep=',', eol='\n')
                            graph_to_adjacency = False

                    elif network_dataset and get_n:
                        if graph_name == "netscience":
                            csvfile = f"netscience_unweighted.csv"
                            data = pd.read_csv(csvfile, header = None)
                            data = data.values
                            # print(f"Shape of adjacency matrix: {data.shape}")
                            Graph = igraph.Graph.Adjacency(data.astype(bool).tolist(), mode = "undirected")
                            G = Graph.connected_components().giant()
                            del data
                        elif graph_name in facebook_networks:
                            csvfile = f"{graph_type}_matrix.csv"
                            data = pd.read_csv(csvfile, header = None)
                            data = data.values
                            # print(f"Shape of adjacency matrix: {data.shape}")
                            Graph = igraph.Graph.Adjacency(data.astype(bool).tolist(), mode = "undirected")
                            G = Graph.connected_components().giant()
                            del data
                        print(graph_name)
                        print('Original number of vertices:', G.vcount())

                        component_membership = G.clusters(mode='strong').membership
                        sizes = G.clusters(mode='strong').sizes()
                        index_max = sizes.index(max(sizes))
                        component = np.nonzero(np.array(component_membership)==index_max)[0]
                        G = G.induced_subgraph(component)
                        print('GCC number of vertices:', G.vcount())
                        
                        #Get the graph size and minor cluster threshold
                        n = G.vcount()
                        #For cluster sizes <= minor threshold, we consider them minor clusters. Otherwise we consider them major clusters
                        minor_threshold = math.floor(n * minor_clusters_percent/100)  
                        print('\nn = ', n, file=f, flush=True)
                        print('The minor threshold is: ', minor_threshold, file=f, flush=True)
                        get_n = False
                        
                    elif graph_type == "degree-regular":
                        def adjacent_edges(nodes, halfk): 
                            n = len(nodes)
                            #nodes = list(range(n))
                            for i,u in enumerate(nodes):
                                for j in range(i+1,i+halfk+1): 
                                    v = nodes[j % n]
                                    yield u, v
                        def make_degree_regular(n,k):
                            nodes = range(n)
                            G = igraph.Graph()
                        
                            # add vertices to the graph
                            G.add_vertices(nodes)
                            
                            # add edges to the graph
                            G.add_edges(adjacent_edges(nodes,k//2))
                            return G
                            G = make_ring_lattice(n,k)
                     
                    for c in cs:
                        for z in zs:
                            for mu in mus:
                                #indent level (+/- two indents)
                                for opinion_set in opinion_sets:
                            #Default name for matfile if we didn't continue the simulation
                                    if graph_type == "erdos-renyi" and not network_dataset:
                                         matfile = (
                                                    f"{experiment}/matfiles/p{p}"
                                                    f"/delta{delta}-gamma{gamma}"
                                                    f"/p{p}-graph{graph_number}--delta{delta}-gamma{gamma}"
                                                    f"--c{c}--z{z}-mu{mu}-op{opinion_set}.mat"
                                                   )
                                    elif graph_type == "SBM" and not network_dataset:
                                        matfile = (
                                                    f"{experiment}/matfiles/p_aa{p_aa}-p_bb{p_bb}-p_ab{p_ab}"
                                                    f"/graph{graph_number}/delta{delta}-gamma{gamma}"
                                                    f"/graph{graph_number}--delta{delta}-gamma{gamma}"
                                                    f"--c{c}-op{opinion_set}.mat"
                                                  )
                                    elif graph_type == "degree-regular":
                                        matfile = (
                                                    f"{experiment}/matfiles/k{k}/delta{delta}-gamma{gamma}"
                                            f"/k{k}--delta{delta}-gamma{gamma}--c{c}--z{z}-mu{mu}-op{opinion_set}.mat"
                                                  )
                                    else:
                                        matfile = (
                                            f"{experiment}/matfiles/delta{delta}-gamma{gamma}"
                                            f"/delta{delta}-gamma{gamma}--c{c}--z{z}-mu{mu}-op{opinion_set}.mat"
                                                   )
                                    
                                    try:
                                        counter += 1
                                        results = io.loadmat(matfile)
                                        #Parse the results into a dataframe row and add to df
                                        row = generate_row(results, minor_threshold, G)
                                #if we have a random graph number, add that to the row
                                        if not network_dataset and folder_name_dict[graph_type] in random_graph_folders:
                                            row.insert(loc = 0, column = 'graph', value = [graph_number])
                                #append our row to the dataframe
                                        df = pd.concat([df, row], ignore_index=True)

                                # break
                                        if row['bailout'][0]:
                                            print('Bailout time reached for ' +  matfile, file=f, flush=True)
                                            print('Bailout time was ' + str(results['T'][0][0]), file=f, flush=True)
                                            print('Max opinion cluster diameter: ', row['max_diameter'][0], file=f, flush=True)
                                            total_bailout += 1
                                    except FileNotFoundError as e:
                                        if counter < 5:
                                            print("File not found:", e)
                                    except TypeError as e:
                                        if counter < 5:
                                            print("Type error in generate_row:", e)
                                    except NameError as e:
                                        if counter < 5:
                                            print("Name error in generate_row:", e)
                                    except AttributeError as e:
                                        if counter < 5:
                                            print("Attribute error in generate_row:", e)
                                    except:
                                        if counter < 3:
                                            print("ahhhhh")
                                            raise
                                            print("file does not exist: ", matfile, file=f, flush=True)
                                        file_not_found += 1

                #Make sure the opinion_set, n_clusters, and time steps are integers instead of floats
                df = df.astype({'opinion_set': int, 'T': int})

                #Check that a directory exists, and if not, create it
                directory = experiment + '/combined_results'
                if not network_dataset:
                    if graph_type == "erdos-renyi":
                        directory = directory + '/p' + str(p)
                    elif graph_type == "SBM":
                        directory = directory + f"/p_aa{p_aa}-p_bb{p_bb}-p_ab{p_ab}"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                filename = (directory + '/delta' + str(delta) + '-gamma' + str(gamma) 
                            + (("--tol" + str(tol)) * toltest) + '.csv')

                df.to_csv(filename, index=False, header=True)

                print('Total bailout reached to rerun:', total_bailout, file=f, flush=True)
                print('Number of files not found:', str(file_not_found), file=f, flush=True)