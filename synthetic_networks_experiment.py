import numpy as np
import pandas as pd
from scipy import io
import sys
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import time as time
import igraph as igraph
import os.path
from os import getpid
import multiprocessing

# Make process "nicer" and lower priority
import psutil
psutil.Process().nice(1)# if on *ux

# Import our own adaptive-confidence bounded-confidence module
import sys
# sys.path.append('..') #look one directory above

import DW as DW


class edge_weight_DW:
    #Set class parameters
    tol = .01 #1e-6 #Diameter required for convergence critera of opinion clusters
    Tmax = 10**9 #Bailout time for ending the simulation
    
    print("I've made a class!!")
    
    #Folder names by graph type for saving outputs
    folder_names = {'complete': 'Complete', 'erdos-renyi': 'Erdos-Renyi', 'degree-regular': 'Degree-Regular'}
    
    random_graph_types = ["complete", "erdos-renyi"]

    def __init__(self, graph_type, n, p=False):
        """
        blah blah 
        """
        #defining instance variables
        print("I'm initializing")
        self.graph_type = graph_type
        self.n = n

        
        self.foldername = f"{graph_type}{n}" #savefolder name for experiment
         
        #Check that a directory for this experiment exists, and if not, create it
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
            os.makedirs(self.foldername + '/matfiles')
            os.makedirs(self.foldername + '/txtfiles')


        if self.graph_type == "erdos-renyi":
            self.p = p
        
        # if self.graph_type == "degree-regular":
        #     self.k = k
        #     if not os.path.exists(f"{self.foldername}/matfiles/k{self.k}"):
        #         os.makedirs(f"{self.foldername}/matfiles/k{self.k}")
            
        #     if not os.path.exists(f"{self.foldername}/txtfiles/k{self.k}"):
        #         os.makedirs(f"{self.foldername}/txtfiles/k{self.k}")
        if self.graph_type == "degree-regular":
            self.k = k
   

    

    
    def generate_seed_files(self):
        '''
        Generate and save random seed files for random graphs (if not complete) and initial opinions if don't exist yet
        '''
        
        self.graph_seed_file = f"{self.graph_type}{n}/graph_seeds.csv"
        self.opinion_seed_file = f"{self.graph_type}{n}/opinion_seeds.csv"
  
        if self.graph_type == "complete":
            #There is only one opinion seed for a complete graph, so we generate and save it if it doesn't exist yet
            if not os.path.exists(self.opinion_seed_file):
                df = pd.DataFrame(columns = ['opinion_seed'])
                random.seed(a=None) #reset random by seeding it with the current time
                weight_seed = str(random.randrange(sys.maxsize))
                df.loc[0] = [weight_seed]
                df.to_csv(self.opinion_seed_file, index=False, header=True)
                
        elif self.graph_type == "erdos-renyi":
            
            #There is only graph seed per p value for an erdos-renyi graph, so we generate and save it if it doesn't exist yet
            if not os.path.exists(self.graph_seed_file):
                df = pd.DataFrame(columns = ['p', 'graph_seed'])
                df.to_csv(self.graph_seed_file, index=False, header=True)
            df = pd.read_csv(self.graph_seed_file)
            row = df[df['p'] == self.p]
            if len(row) == 0:
                random.seed(a=None) #reset random by seeding it with the current time
                graph_seed = str(random.randrange(sys.maxsize))
                row = pd.DataFrame(columns = ['p', 'graph_seed'])
                row.loc[0] = [self.p, graph_seed]
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv(self.graph_seed_file, index=False, header=True)
            
            if not os.path.exists(self.opinion_seed_file):
                df = pd.DataFrame(columns = ['p', 'graph', 'opinion_seed'])
                df.to_csv(self.opinion_seed_file, index=False, header=True)

        elif self.graph_type == "degree-regular":
            #There is only one opinion seed for a degree-regular graph, so we generate and save it if it doesn't exist yet
            if not os.path.exists(self.opinion_seed_file):
                df = pd.DataFrame(columns = ['k', 'opinion_seed'])
                df.to_csv(self.opinion_seed_file, index=False, header=True)
            df = pd.read_csv(self.opinion_seed_file)
            row = df[df['k'] == self.k]
            if len(row) == 0:
                random.seed(a=None) #reset random by seeding it with the current time
                opinion_seed = str(random.randrange(sys.maxsize))
                row = pd.DataFrame(columns = ['k', 'opinion_seed'])
                row.loc[0] = [self.k, opinion_seed]
                if df.empty:
                    # If df is empty, assign row to df directly
                    df = row
                    # print(df)
                else:
                    # Concatenate the row with df
                    df = pd.concat([df, row], ignore_index=True)
                    # print(df)
                df.to_csv(self.opinion_seed_file, index=False, header=True)
            
                
         #File where the random seeds for simulation are stored. 
        if self.graph_type == 'complete':
            self.sim_seed_file = self.foldername + '/sim_seeds.csv'
        if self.graph_type == 'erdos-renyi':
            self.sim_seed_file = self.foldername + '/sim_seeds/p-' + str(self.p) + '.csv'
        if self.graph_type =='degree-regular':
            self.sim_seed_file = f"{self.foldername}/sim_seeds.csv"
        
        
        #If sim seed file doesn't already exist, create it
        if not os.path.exists(self.sim_seed_file):
            # df = pd.DataFrame(columns = ['c', 'mu', 'delta', 'gamma','z', 'opinion_set', 'sim_seed'])
            df = pd.DataFrame(columns=['c', 'mu', 'delta', 'gamma', 'z', 'opinion_set', 'sim_seed'], dtype=float)
            if self.graph_type == "erdos-renyi":
                df.insert(0,'graph_number','')
                df.insert(0,'p','')
            if self.graph_type == "degree-regular":
                df.insert(0,'k','')
            df.to_csv(self.sim_seed_file, index=False, header=True)

        return
        
    ## Function to Run DW model for this graph and weight/opinion seeds  
    def run_DW(self, params):
    
        '''
        Runs DW experiment and saves appropriate output files
        Takes in a dictionary params, containing "c" - the initial confidence 
        bound, "mu" - the compromise parameter, "delta" - the edge weight 
        decrease parameter, "gamma" - the edge weight increase parameter, "z" -
        the initial edgeweight, and "opinion_set" - an integer representing 
        which opinion set to generate from the random opinion 
        seed to run the DW model on. 
        For a complete graph, we only need these parameters.
        For Erdos-Renyi graphs, the 5th parameter, "graph_number" needs to be specified,
        and it represents which randomly generated graph to consider.
        '''
        
        print('Process Number ', getpid())
        print('Params', params)

        ## Initial set up
        #Unpack parameters
        c, mu, z = params["c"], params["mu"], params["z"] 
        delta, gamma = params["delta"], params["gamma"]
        opinion_set = params["opinion_set"]
        if self.graph_type == 'erdos-renyi':
            graph_number = params["graph_number"]
        if self.graph_type == 'degree-regular':
            k = params["k"]

        ## Read the random seeds if they exist, and generate and store them if they don't exist yet
        lock.acquire()

        #Make sure the appropriate save folders for this delta-gamma combo exist, and if not, create them
        sub_folder = f"/delta{delta}-gamma{gamma}"
        if self.graph_type == 'complete':
            if not os.path.exists(self.foldername + "/matfiles" + folder):
                os.makedirs(self.foldername + '/matfiles' + folder)
        
        if self.graph_type == 'erdos-renyi':
            if not os.path.exists(f"{self.foldername}/matfiles/p{p}{sub_folder}"):
                os.makedirs(f"{self.foldername}/matfiles/p{p}{sub_folder}")
            
        if self.graph_type == 'degree-regular':
            if not os.path.exists(f"{self.foldername}/matfiles/k{k}{sub_folder}"):
                os.makedirs(f"{self.foldername}/matfiles/k{k}{sub_folder}")
            if not os.path.exists(f"{self.foldername}/txtfiles/k{k}"):
                os.makedirs(f"{self.foldername}/txtfiles/k{k}")


        # Get the random graph seed if not a complete graph
        if self.graph_type == 'erdos-renyi':
            df = pd.read_csv(self.graph_seed_file)
            row = df[df['p'] == self.p]
            graph_seed = row['graph_seed'].values[0]
            graph_seed = int(graph_seed)

    
        # Get the random opinion set seed 
        df = pd.read_csv(self.opinion_seed_file)
        if self.graph_type == 'complete' :
            opinion_seed = df['opinion_seed'].values[0]
            opinion_seed = int(opinion_seed)
        elif self.graph_type == 'degree-regular': 
            row = df[df['k'] == k]
            opinion_seed = row['opinion_seed'].values[0]
            opinion_seed = int(opinion_seed)
        elif self.graph_type == 'erdos-renyi':
            row = df[df['p'] == self.p]
            row = row[row['graph'] == graph_number]
            if len(row) == 0:
                random.seed(a=None) #reset random by seeding it with the current time
                opinion_seed = random.randrange(sys.maxsize)
                row = pd.DataFrame(columns = ['p', 'graph', 'opinion_seed'])
                row.loc[0] = [self.p, graph_number, str(opinion_seed)]
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv(self.opinion_seed_file, index=False, header=True)
            else:
                opinion_seed = row['opinion_seed'].values[0]
                opinion_seed = int(opinion_seed)
        lock.release()
        
        ## Specify the save file names for matfiles and txtfiles 
        savename = ""
        if self.graph_type == 'erdos-renyi':
            savename = f"p{p}-graph{graph_number}"
        if self.graph_type == "degree-regular":
            savename = f"k{k}"
        savename = f"{savename}--delta{delta}-gamma{gamma}--c{c}--z{z}"
        if self.graph_type == 'erdos-renyi':
            txtfile = f"{self.foldername}/txtfiles/p{p}/{savename}.txt"
        if self.graph_type == "degree-regular":
            txtfile = f"{self.foldername}/txtfiles/k{k}/{savename}.txt"
        else:
            txtfile = f"{self.foldername}/txtfiles/{savename}.txt"
        savename = f"{savename}-mu{mu}"
        
        ## If the txtfile doesn't exist yet, create it and write the header with seed values to it
        lock.acquire()
        if not os.path.exists(txtfile):
            print(txtfile)
            try:
                with open(txtfile, 'w') as f:
                    print('Experiment:', self.graph_type, ", n =", self.n, file=f, flush=True)
                    if self.graph_type == 'erdos-renyi':
                        print("p =", self.p, ", graph_number = ", graph_number, file=f, flush=True)
                        print('graph_seed = ', graph_seed, file=f, flush=True)
                    if self.graph_type == "degree-regular":
                        #print("k=", self.k, file=f, flush=True)
                        print("k=", k, file=f, flush=True)
                    print('delta = ', delta, ' and gamma = ', gamma, file=f, flush = True)
                    print('c = ', c, file=f, flush = True)
                    print('z = ', z, file=f, flush = True)
                    print('opinion_seed = ', opinion_seed, file=f, flush = True)
            except: 
                print("In exception:", e)
        lock.release()
        
        # Functions to Make Degree Regular Graph
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
    
        #make graphs 
        if self.graph_type == "complete":
            G = igraph.Graph.Full(self.n)
        elif self.graph_type == 'degree-regular':
            G = make_degree_regular(self.n,k)
        elif self.graph_type == "erdos-renyi":
            #Reinitialize the random seed and generate the corresponding graph number from that seed
            random_graph = np.random.default_rng(graph_seed)
            for i in range(graph_number + 1):
                seed = random_graph.integers(low=0, high=sys.maxsize)
                random.seed(a=seed)
                G = igraph.Graph.Erdos_Renyi(self.n, self.p)
            
        #Reinitialize the random seed and generate the corresponding opinion set from that seed
        random_opinion = np.random.default_rng(opinion_seed)
        for i in range(opinion_set + 1):
            init_opinions = random_opinion.uniform(0, 1, size=self.n)
        G.vs['opinion'] = init_opinions

        ## Read or create a simulation seed for DW node selection for this set of parameters (c, mu, z, weight_set, opinion_set)
        #Read the random seed csv file as a pandas dataframe
        lock.acquire()
        df = pd.read_csv(self.sim_seed_file)
        
        #Try to get the corresponding dataframe row for this simulation
        row = df[df['c'] == c]
        row = row[row['mu'] == mu]
        row = row[row['delta'] == delta]
        row = row[row['gamma'] == gamma]
        row = row[row['z'] == z]
        row = row[row['opinion_set'] == opinion_set]
        if self.graph_type == "erdos_renyi":
            row = row[row['p'] == self.p]
            row = row[row['graph_number'] == graph_number]
        if self.graph_type == "degree-regular":
            row = row[row['k'] == k]
            
        #If there isn't already an entry for this simulation generate and store a simulation seed
        if len(row) == 0:
            random.seed(a=None) #reset random by seeding it with the current time so we don't keep generating the same sim seeds
            sim_seed = random.randrange(sys.maxsize)
            param_dict = {
                'c': [c],
                'mu': [mu],
                'delta': [delta],
                'gamma': [gamma],
                'z': [z],
                'opinion_set': [opinion_set],
                'sim_seed': [str(sim_seed)]
            }
            
            # Create DataFrame from the dictionary
            row = pd.DataFrame(param_dict)
            row.loc[0] = [c, mu, delta, gamma, z, opinion_set, str(sim_seed)]
            if self.graph_type == "erdos-renyi":
                row.insert(0,'graph_number', graph_number)
                row.insert(0,'p', self.p)
            if self.graph_type == 'degree-regular':
                row.insert(0,'k',k)
            try:
                if df.empty:
                    # If df is empty, assign row to df directly
                    df = row
                    # print(df)
                else:
                    # Concatenate the row with df
                    df = pd.concat([df, row], ignore_index=True)
                    # print(df)
            except:
                print(":-(")
                            
            df.to_csv(self.sim_seed_file, index=False, header=True)
        else:
            sim_seed = row['sim_seed'].values[0]
            if len(row) > 1:
                with open(txtfile, 'w') as f:
                    print("OH NO! Something went wrong and there are multiple sim_seeds", file=f, flush=True)
                    print(row, file=f, flush=True)
        lock.release()
            
        #Time the DW simulation for this weight + opinion set combo
        start_time = time.time()

        ## Run the DW model using the simulation seed
        # print('Process Number ', getpid(), 'starting DW with sim_seed = ', sim_seed) #deleteline

        outputs = DW.DW(G, c, mu, z, gamma, delta, random_seed = sim_seed, 
                        tol = self.tol, Tmax = self.Tmax)

        lock.acquire()
        with open(txtfile, 'a') as f:
            print("\n----- mu = %f and opinion_set = %s -----" % (mu, opinion_set), file=f, flush=True)

            print("T = %s" % outputs['T'], file=f, flush=True)
            # print("Min confidence = %.3f, and Max confidence = %.3f" % (min(outputs['confidence']), max(outputs['confidence'])), file=f, flush=True)
            print("Number of Clusters = %s" % outputs['n_clusters'], file=f, flush=True)

            print("Cluster Membership", file=f, flush=True)
            print(outputs['clusters'], file=f, flush=True)
            
            runtime = time.time() - start_time
            print('-- Runtime was %.0f seconds = %.3f hours--' % (runtime, runtime/3600) , file=f, flush=True)
        lock.release()

        ## Define dictionary to store simulation outputs for saving to a .mat file
        save_sim = {'c': c, 'mu': mu, 'delta': delta, 'gamma':gamma, 'z': z, 'opinion_set': opinion_set, 'sim_seed': sim_seed}
        
        #Include the graph-level results
        save_sim['T'] = outputs['T']                    
        save_sim['T_changed'] = outputs['T_changed']
        save_sim['T_acc'] = outputs['T_acc']
        save_sim['bailout'] = outputs['bailout']
        save_sim['avg_opinion_diff'] = outputs['avg_opinion_diff']
        
        #Include the cluster information
        clusters = outputs['clusters']
        save_sim['n_clusters'] = outputs['n_clusters']
        for i in range(outputs['n_clusters']):
            key = 'cluster' + str(i)
            save_sim[key] = clusters[i]
            #clusters can be extracted from matfile using list = clusteri.flatten().tolist()
            
        #Include the node-level results as size n arrays
        save_sim['init_opinions'] = init_opinions
        save_sim['final_opinions'] = outputs['final_opinions']
        save_sim['edge_weights'] = outputs['edge_weights']
        save_sim['total_change'] = outputs['total_change']
        save_sim['n_updates'] = outputs['n_updates']
        save_sim['local_receptiveness'] = outputs['local_receptiveness']

        

        ## Save the simulation results to a matfile
        if self.graph_type == "erdos-renyi":
            matfile = f"{self.foldername}/matfiles/p{p}{sub_folder}/{savename}-op{opinion_set}.mat"
        if self.graph_type == 'degree-regular':
            matfile = f"{self.foldername}/matfiles/k{k}{sub_folder}/{savename}-op{opinion_set).mat"
        else:
            matfile = f"{self.foldername}/matfiles/{sub_folder}/{savename}-op{opinion_set}.mat"
        io.savemat(matfile, save_sim)
        
        

def init(l):
    global lock
    lock = l

if __name__ == "__main__":
## EXPERIMENT PARAMETERS - CHANGE HERE
    graph_type = 'degree-regular'
    n = 1000 # Complete graph size
    # ks = [2,4,10,50,100,200,300] 
    ks = [20] 

    # #baseline
    #gammas = [0.0] #Confidence-increase parameters
    #deltas = [1.0] #Confidence-decrease parameters

    #other gamma/delta values
    gammas = [0.1, 0.5, 0.9] #Confidence-increase parameters
    deltas = [0.1, 0.5, 0.9] #Confidence-decrease parameters

    
    zs = [0.1, 0.5, 0.9]
    
    cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #Initial confidence bound
    #cs = [0.1, 0.2, 0.7, 0.9] 
    mus = [0.1, 0.3, 0.5] #Compromise parameter

    opinion_sets = list(range(0,12)) #Which opinion sets to run


    # #test parameters
    # gammas = [0.3, 0.5] #Confidence-increase parameters
    # deltas = [0.5,0.9] #Confidence-decrease parameters
    
    # zs = [0.5]
    
    # cs = [0.4] #Initial confidence bound
    # mus = [0.5] #Compromise parameter
    
    # opinion_sets = list(range(0,1)) #Which opinion sets to run
    # print("compiling parameters list")
    
    ## Generate list of tuples to feed into DW_experiments as parameters
    params_list = []
    for graph_number in graph_numbers: 
        for k in ks:
            for delta in deltas:
                for gamma in gammas:
                    for c in cs:
                        for mu in mus:
                            for z in zs: 
                                for opinion_set in opinion_sets:
                                    if graph_type == "degree-regular":
                                        matfile = (f"degree-regular{n}/matfiles/k{k}/delta{delta}-gamma{gamma}/"
                                                       f"k{k}--delta{delta}-gamma{gamma}--c{c}--z{z}-mu{mu}-op{opinion_set}.mat")
                                    elif graph_type == "erdos-renyi":
                                        matfile = (f"erdos-renyi{n}_p{p}/matfiles/p{p}/delta{delta}-gamma{gamma}/p{p}-graph{graph_number}"
                                               f"--delta{delta}-gamma{gamma}--c{c}--z{z}-mu{mu}-op{opinion_set}.mat")
                                    else:
                                        matfile = (f"{graph_type}{n}/matfiles/delta{delta}-gamma{gamma}/"
                                                   f"delta{delta}-gamma{gamma}--c{c}--z{z}-mu{mu}-op{opinion_set}.mat")
                                    try:
                                        results = io.loadmat(matfile)
                                            
                                    except:
                                        param_dict = {"delta": delta, "gamma": gamma,
                                                        "c": c, "mu": mu, "z": z,
                                                        "opinion_set": opinion_set}
                                        if graph_type == "erdos-renyi":
                                            param_dict["graph_number"] = graph_number
                                        if graph_type == "degree-regular":
                                            param_dict["k"] = k
                                        params_list.append(param_dict)  
    ## adding in baseline parameters
    for graph_number in graph_numbers: 
        for k in ks:
            for c in cs:
                for mu in mus:
                    for opinion_set in opinion_sets:
                        if graph_type == "erdos-renyi":
                            matfile = (f"erdos-renyi{n}/matfiles/p{p}/delta{1}-gamma{0}/p{p}-graph{graph_number}"
                                       f"--delta{1}-gamma{0}--c{c}--z{0.5}-mu{mu}-op{opinion_set}.mat")
                        if graph_type == "degree-regular":
                            matfile = (f"degree-regular{n}/matfiles/k{k}/delta{1}-gamma{0}/"
                                       f"k{k}--delta{1}-gamma{0}--c{c}--z{0.5}-mu{mu}-op{opinion_set}.mat")
                        else:
                            matfile = (f"{graph_type}{n}/matfiles/delta{delta}-gamma{gamma}/"
                                       f"delta{delta}-gamma{gamma}--c{c}--z{z}-mu{mu}-op{opinion_set}.mat")
                        try:
                            results = io.loadmat(matfile)
                            
                        except:
                            param_dict = {"delta": 1, "gamma": 0,
                                          "c": c, "mu": mu, "z": 0.5,
                                          "opinion_set": opinion_set}
                            if graph_type == "erdos-renyi":
                                param_dict["graph_number"] = graph_number
                            if graph_type == "degree-regular":
                                param_dict["k"] = k
                            params_list.append(param_dict) 
    #print(params_list)                
    #Initialize experiment class
    experiment = edge_weight_DW('degree-regular', n = n, p=False)
    experiment.generate_seed_files()

    l = multiprocessing.Lock()

    with multiprocessing.Pool(processes=75, initializer=init, initargs=(l,)) as pool:
        pool.map(experiment.run_DW, params_list)