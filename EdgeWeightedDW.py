
import numpy as np
import pandas as pd
from scipy import io, stats
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import time as time
import igraph as igraph
import sys

#Define colormaps for matrix plotting functions
OFFDIAG_CMAP = mpl.colors.LinearSegmentedColormap.from_list("mycmap", ['0.1', '0.9'])
DIAG_CMAP = 'seismic'


#DW model implemented in igraph with an edge weights that encode edge probabilities
def DW(graph, d, mu, z, gamma, delta, tol = 10**-5, t_start = 0, z_start = None, Tmax = 10**8, Tmin = 0, random_seed = None, T_acc = None, sigfigs = 3,
       return_opinion_series = False, opinion_timestep = 1, return_matrix = False, matrix_timestep = 100, nodeorder = None):
    
    """
    Simulates the Deffaunt--Weisbuch model of opinion dynamics.
    Uses and ego-centric approach to sort nodes into clusters and check convergence
    
    Parameters
    ------------
    graph : igraph.graph
        igraph Graph with 'opinion' node-level attributes and n nodes
    d : float
        Confidence bound (homogeneous)
        NOTE: THE CONFIDENCE BOUND IS CALLED d HERE, BUT c IN THE PAPER
    mu : float
        Compromise parameter, also known as convergence or cautiousness parameter (homogeneous)
        NOTE: THE COMPROMISE PARAMETER IS CALLED mu HERE, BUT m IN THE PAPER
    z : float (could change to vector later if we want to consider non-uniform initial weights)
        Initial edge weights. 
    gamma : float, in [0,1]
        edge-weight increase parameter
    delta : float, in [0,1]
        edge-weight decrease parameter
    tol : float, default = 0.02
        The tolerence/convergence critera. All opinion clusters must have difference between 
        maximum and minimum opinion less that tol for the simulation to be converged
    t_start : int, default = 0
        What timestep to start counting the model at. By default t = 0 and the model is started at the "beginning."
        To continue a simulation, set t_start to the time left off
    Tmax : int, default = 10**8
        Maximum number of timesteps for the simulation.
    Tmin : int, default = 0
        Minimum number of timesteps before we start checking convergence.
    random_seed: float, optional
        random seed for node selection at each timestep of the model
    T_acc : int, default = None
        Accuracy in convergence time.
        This is the number of timesteps we advance the model before checking for convergence again.
        The default value is to use an adaptive T_acc based on the sigfigs parameter
    sigfigs : int, default = 3
        The number of significant digits of accuracy for T. The accuracy in the number of decimal
        places for ln(T) is sigfigs - 1. This paramter controls the adaptive change of T_acc, and
        therefore the tradeoff between runtime and accuracy in T
        
    Other Parameters
    ----------------
    return_opinion_series : bool, optional
        If True, returns matrix containing each node's opinion at each time
    opinion_timestep: int, default = 1
        If return_opinion_series is true, the opinion values will be added to the opinion time series
        every opinion_timesteps steps
    return_matrix : bool, optional
        If True returns an array of adjacency matrices for visualization
    matrix_timestep : int, default = 10
        If return_matrix is true, a matrix will be generated every matrix_timestep steps
    nodeorder : int array, optional
        If return_matrix is true, nodeorder, a list of values 0 to n-1 specifies the row/col
        order of returned visualization matrices
    
    Returns
    ----------------
    output : dictionary
        Dictionary of simulation outputs of {T: int, T_changed: int, n_clusters: int, 
        clusters: list of lists, final_opinions: list of float, total_change: list of float}
        If return_opinion_series is True, also includes {opinion_series: numpy.array}
        If return_matrix is True, also includes {visualization_matrix: list}
        
        <T> int : number of timesteps to converge
        <T_changed> int : number of timesteps in which opinions changed
        <T_acc> int : how accurate T, the number of timesteps is
        <bailout> bool: True if bailout time was reached before final convergence.
                        Opinion cluster membership may still be reported if initial convergence check passed.
                        
        <n_clusters> int : number of opinion clusters
        <clusters> list of list: cluster membership lists by int node ID
        
        <final_opinions> list of float : the final opinions of each node
        <total_change> list of float: total absolute distance each node changed opinion
        <n_updates> list of float: total number of times each node updated its opinion value
        <local_agreement> list of float: local agreement for each node, this is the fraction of neighboring
                            nodes that are on the same side of the mean
        <local_receptiveness> list of float: local receptiveness for each node, this is the fraction of neighboring
                            nodes that are within the confidence bound, and therefore that the node is willing to interact with
        
        <opinion_series> numpy.array : n x T matrix of each node's opinion at each time
        <matrix_list> list of numpy.array: list of visualization matrices generated by get_visualization_matrix

        <random_seed> float : random seed used for random selection of nodes at each timestep  
     """
    
    #Make a copy of the graph
    G = graph.copy()
    
    n = G.vcount() #get number of nodes/verticies in the network
    
    ## Get the edge information for the graph
    # Store the edge nodes and values (maginutide of opinion difference across the edge) as numpy arrays
    edge_start = np.array([e.tuple[0] for e in G.es], dtype=np.int64)
    edge_end = np.array([e.tuple[1] for e in G.es], dtype=np.int64)
    edge_value = np.array(G.vs[edge_start.tolist()]['opinion']) - np.array(G.vs[edge_end.tolist()]['opinion'])
    edge_value = np.abs(edge_value) #the edge value is the magnitude of the opinion difference between the two nodes
    
    #If no T_acc was provided, we try and make it adaptive to the provided sig-figs
    #Otherwise, we use the provided T_acc value
    adaptive_T_acc = False
    if T_acc == None:
        adaptive_T_acc = True
        T_acc = 0 #initialize T_acc to 0, we want the exact number of time steps to start
    
    # Set random seed if given, otherwise generate, use and store a new seed
    if random_seed == None:
        random_seed = random.randrange(sys.maxsize)
    random.seed(a=random_seed)
    
    #If we are continuing from another simulation, get to the right time point drawing random numbers with this seed
    for i in range(t_start):
        number = random.uniform(0,1)

    ### initialize edge weight
    if z_start is not None:
        edge_weights = z_start
    else:
        edge_weights = z*np.ones_like(edge_start)

    
    # # So we only do the calculation once, 
    # # we get the probabilities and cumulative probability vector to pick the first node
    # # Normalize the weights to create probabilities for each node
    # weights = G.vs['weight']
    # weights = np.array(weights)
    # p_activate = weights / sum(weights)
    # # Get vector of cumulative probabilities
    # p_cumulative = np.cumsum(p_activate)
    
    # Store initial opinions if return_opinion_series is True
    if return_opinion_series:
        opinion_series = G.vs['opinion']
    
    # Store visualization matrices if return_matrix is True
    if return_matrix:
        M = get_visualization_matrix(G, d, nodeorder = nodeorder)
        matrix_list = [M]
    
    # Initialize a list of total opinion change and number of update times for each node
    total_change = [0] * n
    n_updates = [0] * n
    
       
    # Keep track of how many time steps actual resulted in an opinion change
    T_changed = 0
    
    # Keep track of if we passed the initial convergence checks and final convergence criteria
    # First check = all nodes are <= tol or > d in opinion with their neighbors
    # Second check = nodes are in distinct opinion clusters separated by at least d
    # Final convergence = all clusters (already set in second check) have diameter < tol
    first_check_passed = False
    second_check_passed = False
    converged = False
    
    # Track if opinions changed at all and number of timesteps since last convergence check
    # We only check for convergence if it has been T_acc timesteps since the last check
    # and the opinions have changed at least once in those timesteps
    step_counter = 0
    opinions_changed = False
    
    ### RUN THE DEFFAUNT-WEISBUCH MODEL UNTIL TMAX AT MOST
    for t in range(t_start + 1, Tmax+1):
    
        # Update timestep counter for convergence checking
        step_counter = step_counter + 1

        ### TODO: CHANGE! 
        ### RANDOMLY GET NEXT EDGE TO INTERACT
        ## For each time step, pick an edge based on edge weight
        p_cumulative = np.cumsum(edge_weights)
        rand = random.uniform(0,np.sum(edge_weights))
        edge_index = np.argmax(p_cumulative > rand)

        node_1 = edge_start[edge_index]
        node_2 = edge_end[edge_index]
        old_edge_weight = edge_weights[edge_index]

        # print(str(node_1) + " and " + str(node_2))

        opinion1 = G.vs[node_1]['opinion']
        opinion2 = G.vs[node_2]['opinion']

        if ( abs(opinion1 - opinion2) < d ):
            #update the node's opinions
            G.vs[node_1]['opinion'] = opinion1 + mu*(opinion2 - opinion1)
            G.vs[node_2]['opinion'] = opinion2 + mu*(opinion1 - opinion2)

            new_edge_weight = old_edge_weight + gamma*(1-old_edge_weight)
            edge_weights[edge_index] = new_edge_weight
            
            #update our variables for the timesteps with opinion changes, and total opinion change
            T_changed = T_changed + 1
            change = mu*abs(opinion2 - opinion1)
            total_change[node_1] = total_change[node_1] + change
            total_change[node_2] = total_change[node_2] + change
            n_updates[node_1] = n_updates[node_1] + 1
            n_updates[node_2] = n_updates[node_2] + 1
            opinions_changed = True
            
            #update the edge values of the opinion differences with the two nodes
            index = np.argwhere((edge_start == node_1) | (edge_start == node_2) | (edge_end == node_1) | (edge_end == node_2))
            index = index.flatten()
            start_node, end_node = edge_start[index].tolist(), edge_end[index].tolist()
            edge_value[index] = np.abs( np.array(G.vs[start_node]['opinion']) - np.array(G.vs[end_node]['opinion']) )
        else: 
            new_edge_weight = delta*old_edge_weight
            edge_weights[edge_index] = new_edge_weight
            
 
        
        # Store the opinions at the end of this time step if needed
        if return_opinion_series and (t % opinion_timestep == 0):
            opinion_series = np.vstack((opinion_series, np.asarray(G.vs['opinion'])))
        
        # Generate and store the visualization matrix at the end of this time step if needed
        if return_matrix and (t % matrix_timestep == 0):
            M = get_visualization_matrix(G, d, nodeorder = nodeorder)
            matrix_list.append(M)

        ## CHECK CONVERGENCE
        # Proceed to checking convergence it has been T_acc time steps since the last convergence check
        # AND opinions have changed at least once in those timesteps
        if (opinions_changed == True) and (step_counter >= T_acc) and (t > Tmin):
            
            # Adapt and increase T_acc by a factor of 10 if the timesteps has increased by a factor of 10
            if adaptive_T_acc:
                digits = math.floor(math.log10(t))
                T_acc = 10**(digits - sigfigs)

            # Reset timestep counter and opinion change tracking
            step_counter = 0
            opinions_changed = False
            
            ### FIRST CONVERGENCE CHECK
            ## Each node must have neighbors that are either < tol away, or >= d away
            if not first_check_passed:
                
                first_check_passed = True
                index = np.argwhere((tol <= edge_value) & (edge_value < d))
                index = index.flatten()                
                if len(index) > 0:
                    first_check_passed = False

            ### SECOND CONVERGENCE CHECK
            ## Nodes must be in distinct clusters (distance between clusters >= d)
            ## And check cluster must have diameter < d
            if first_check_passed and (not second_check_passed):
                
                second_check_passed = True
                
                ## Construct the effective subgraph, which only consists of edges with
                ## magnitude of the opinion difference less than d, i.e. the edges of possible interactions

                #Get the edges which have opinion difference < d and convert them to an edge list
                index = np.argwhere(edge_value < d)
                index = index.flatten()
                sub_edge_start = edge_start[index]
                sub_edge_end = edge_end[index]
                edge_list = [(sub_edge_start[i], sub_edge_end[i]) for i in range(len(index))]

                #Construct an effective subgraph subG consisting of these edges with possible interactions 
                subG = igraph.Graph()
                subG.add_vertices(n)
                subG.add_edges(edge_list)
                subG.vs['opinion'] = G.vs['opinion']

                #Get the opinion clusters, the components of the subgraph
                cluster_membership = subG.clusters(mode='strong').membership
                n_clusters = max(cluster_membership)+1
                clusters = []
                for i in range(n_clusters):
                    cluster = np.nonzero(np.array(cluster_membership)==i)[0]
                    clusters.append(cluster.tolist())

                ## Check if the constructed clusters all have diameter < d:
                for cluster in clusters:
                    opinions = G.vs[cluster]['opinion']
                    diameter = max(opinions) - min(opinions)
                    if diameter >= d:
                        second_check_passed = False 
                        break

            ### FINAL CONVERGENCE CHECK
            ## Nodes must be in distinct clusters (from passing the second check)
            ## And each cluster must have diameter < tol
            if second_check_passed:
                converged = True
                for cluster in clusters:
                    opinions = G.vs[cluster]['opinion']
                    diameter = max(opinions) - min(opinions)
                    if diameter >= tol:
                        converged = False 
                        break
                
        ## If we're converged, stop running the model
        if converged:
            break
    
    ## Now that we are done running the DW simulation, return results
    
    T = t #get the convergence time
    final_opinions = G.vs['opinion'] #store the final opinions
    
    # If we didn't pass the second convergence check of distinct clusters of diameter < d
    # Do not return any clusters
    if not second_check_passed:
        clusters = []
    
    # If we hit Tmax and were not converged, specify that we hit the bailout time
    # Add 9 to T just to distinguish that actual convergernce takes > Tmax time
    bailout = False
    if T == Tmax and (not converged):     
        bailout = True
    
    n_clusters = len(clusters) # calculate the number of cluster
    
    #Calculate the local agreement for each node
    G.vs['sign'] = np.sign(final_opinions - np.mean(final_opinions))
    local_agreement = [( np.sum(G.vs[G.neighbors(node)]['sign'] == G.vs[node]['sign']) / G.degree(node) ) for node in range(n)]
    
    #Calculate the local receptiveness for each node - this is the fraction of neighbors that are < d away in opinion
    if not first_check_passed:
        #Get the edges which have opinion difference < d and convert them to an edge list
        index = np.argwhere(edge_value < d)
        index = index.flatten()
        sub_edge_start = edge_start[index]
        sub_edge_end = edge_end[index]
        edge_list = [(sub_edge_start[i], sub_edge_end[i]) for i in range(len(index))]

        #Construct an effective subgraph subG consisting of these edges with possible interactions 
        subG = igraph.Graph()
        subG.add_vertices(n)
        subG.add_edges(edge_list)
        
    local_receptiveness = [ (subG.degree(node) / G.degree(node)) for node in range(n) if G.degree(node) > 0 ]
    
    outputs = {}
    outputs['T'] = T
    outputs['T_changed'] = T_changed
    outputs['T_acc'] = T_acc
    outputs['bailout'] = bailout
    outputs['n_clusters'] = n_clusters
    outputs['clusters'] = clusters
    outputs['total_change'] = total_change
    outputs['n_updates'] = n_updates
    outputs['final_opinions'] = final_opinions
    outputs['random_seed'] = random_seed
    outputs['avg_opinion_diff'] = np.sum(edge_value) / len(edge_value)
    outputs['local_agreement'] = local_agreement
    outputs['local_receptiveness'] = local_receptiveness
    outputs['edge_weights'] = edge_weights
    
    if return_opinion_series == True:
        outputs['opinion_series'] = opinion_series
    if return_matrix == True:
        outputs['matrix_list'] = matrix_list

    return outputs

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

#Function to plot opinion_series from DW simulation
def opinion_plot(G, opinion_series, cbar = True, fig=None, ax=None, savefile = None, title = None, fontsizes = {'S': 15, 'M': 23, 'L':25}):
    
    """
    Function to plot the opinion trajectories after DW simulation.
    
    Parameters
    ----------
    G : igraph.graph
        igraph object with 'opinion' and 'weight' node-level attributes and n nodes
    opinion_series : np.array
        Time series output of DW function. T x n matrix where T is number of timesteps and n is number of nodes
        ij-th entry is the opinion at time i of node j
    cbar : bool, default = True
        Boolean value indicating if there should be a colorbar added to the plot and if the trajacetory colors
        should come from node weights/activation probability. If cbar = False, then random rainbow colors will
        be used for the trajectories and there will be no colorbar.
    title : string, optional
        Title for figure
    savefile : string, optional
        File name to save plot
    
    Other Parameters
    ----------------
    fig, ax: matplotlib fig and ax, optional
        If provided (i.e. subplots of a larger plot), then plot will be generated on them
    fontsizes : dictionary
        Dictionary with keys 'S', 'M', 'L' of fontsizes for plot
    
    Returns
    ----------------
    Shows (and saves if savefile is specified) plot of opinion_series opinion trajectories of each node
        
    """
    
    n = G.vcount() #get number of nodes/verticies in the network
    
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize = (9, 6))
    
    #Calculate activation probability of each node as the first of an interaction pair
    weights = G.vs['weight']
    weights = np.array(weights)
    p_activate = weights / sum(weights)
    
    # Set node colors based on weight magnitude
    colormap = mpl.cm.get_cmap('rainbow', 100)
    norm = mpl.colors.Normalize(vmin=0, vmax = round_decimals_up(np.amax(p_activate), decimals=2) )

    # Plot the colored opinion_series, use color based on weights if nonconstant weights
    if cbar: 
        for node in np.argsort(p_activate):
            color = norm(p_activate[node])
            ax.plot(range(0, opinion_series.shape[0]), opinion_series[:,node], color = colormap(color))
    # Plot the colored opinion_series, use rainbow/spaced colors if uniform weights
    else:
        for node in range(0,n):
            ax.plot(range(0, opinion_series.shape[0]), opinion_series[:,node])
            
    ax.set_xlabel("t", fontsize=fontsizes['M'])
    ax.set_ylabel("opinion", fontsize=fontsizes['M'])
    ax.tick_params(axis='both', which='major', labelsize=fontsizes['S'])
    if title:
        ax.set_title(title, fontsize=fontsizes['L'])
    
    # Add color bar if cbar = True
    if cbar: 
        #psm = ax.pcolormesh(data, cmap=colormap, rasterized=True, vmin=-4, vmax=4)
        norm = mpl.colors.Normalize(vmin=0, vmax = round_decimals_up(np.amax(p_activate), decimals=2) )
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
        cbar.ax.tick_params(labelsize=fontsizes['S']) 
        cbar.ax.set_ylabel('node weight', fontsize=fontsizes['M'], rotation = 270, labelpad=fontsizes['M'])
        
    #save file if specified
    if savefile:
        plt.savefig(savefile, bbox_inches='tight', facecolor='white', transparent = False)




# Function to generate visualization matrix of effective subgraph given a graph with opinions
def get_visualization_matrix(graph, d, nodeorder = None):
    """Outputs visualization matrix for an igraph structure with opinion as a node-attribute
    
    Parameters
    ------------
    graph : igraph.Graph
        igraph Graph with 'opinion' node-level attribute and n nodes
    d : float
        Confidence bound. Nodes with absolute opinion difference > d are not connected
        in the effective subgraph
        NOTE: THE CONFIDENCE BOUND IS CALLED d HERE, BUT c IN THE PAPER
    nodeorder : int array, optional
        Array of the ordering of nodes in generated matrix. Default: nodes are ordered 0 to n-1
    
    Returns
    -------
    M : numpy.ndarray
        numpy matrix for visualization of size n x n where nodes are renumbered by nodeorder.
        The diagonal values are the opinions of nodes, so M_ii = opinion(i).
        The off diagonal values are the difference in opinion from the diagonal node, 
        so M_ij = opinion(j) - opinion(i) if i != j
    """
    
    G = graph #copy the graph
    
    n = G.vcount() #get number of nodes/verticies in the network
    
    #If no nodeorder is provided, order from 0 to n-1
    if nodeorder is None:
        nodeorder = list(range(n))
        
    #Add order as a node attribute in the graph
    #G[i]['index'] is the index of node i in nodeorder
    G.vs['index'] = np.argsort(np.array(nodeorder))
        
    # Initialize effective subgraph with reordered nodes
    sub_G = igraph.Graph()
    sub_G.add_vertices(n)
    sub_G.vs['opinion'] = G.vs[nodeorder]['opinion']
    
    # Add edges with "diff" attribute for difference of opinion
    # Only include edges with weight < d, the confidence bound
    for index in range(0,n):
        opinion = sub_G.vs[index]['opinion']
        
        # Get the corresponding not reordered node label in G
        node = nodeorder[index]
        
        # Get the neighborhood in the original/full graph structure
        nbhd = G.neighbors(node)
        
        #Convert the neighborhood to the reordered indices
        nbhd_idx = np.asarray(G.vs[nbhd]['index'])
        
        # For each index, we only set relevent difference in opinion for nodes with larger index
        # That way we don't repeat differences we've already checked
        nbhd_idx = nbhd_idx[np.where(nbhd_idx > index)]
        
        #For each of the remaining neighbors, see if the opinion difference is within d
        for neighbor in nbhd_idx:
            diff = abs(sub_G.vs[neighbor]['opinion'] - opinion)
            if diff < d:
                #if diff = 0, then assign it machine epsilon so it doesn't get masked when plotting
                if diff == 0:
                    diff = np.finfo(float).eps
                sub_G.add_edges([(index, neighbor)], {'diff': diff})
    
    # Get upper triangle of weighted adjacency matrix from the effective subgraph
    #M = sub_G.get_adjacency(type = 'GET_ADJACENCY_UPPER', attribute = 'diff')
    M = sub_G.get_adjacency(attribute = 'diff')
    M = np.asarray(M.data)
    M = np.triu(M, k=1)
    
    # Add in the diagonal as the opinions of the nodes
    diag = np.diag(sub_G.vs['opinion'])
    
    # Add in the lower triangular part, M^T = M (symmetric)
    M = M + np.transpose(M) + diag
    return M

def plot_matrix(matrix, d, savefile = None,
                row_labels = False, col_labels = False, tickspace = 1,
                show_values = False, decimal_places = 2, 
                figsize = (12,10), title = None, fontsizes = {'small': 10, 'medium': 12, 'large':20}):
    
    """
    Plots heatmap of visualization matrix of effective subgraph
    
    Parameters
    ------------
    matrix : numpy.ndarray
        Adjacency matrix of effective subgraph to visualize. Output of get_visualization_matrix.
    d : float
        Confidence bound, which sets the max value of colormap for off diagonal entries
        NOTE: THE CONFIDENCE BOUND IS CALLED d HERE, BUT c IN THE PAPER
    savefile : string, optional
        File name to save plot
    
    Other Parameters
    ----------------
    row_labels, col_labels : array_like, optional
        Row labels and column labels
    tickspace : int, default = 1
        The spacing/frequency of x and y axis row/col labels if using
    show_values : bool or string, default = False
        If True, print values of each nonzero matrix value in the heatmap
        If 'diagonal' then only print values on the diagonal
    decimal_places : int, default = 2
        Number of decimal places to print for matrix values if show_values = True
        Also the number of decimal places for the colorbar of off-diagonal entries
    figsize : tuple, default = (15,10)
        Tuple with two values giving figure dimensions in inches
    title : string, optional
        Title for plot
    fontsizes : dictionary
        Dictionary with keys 'small', 'medium', 'large' of fontsizes for plot
    
    Returns
    -------
    Shows (and saves if savefile is specified) plot of visualization matrix
    
    """
    
    # Create new plot
    fig, ax = plt.subplots(figsize = figsize, facecolor='white')
    
    # Define colormaps
    offdiag_cmap = OFFDIAG_CMAP
    diag_cmap = DIAG_CMAP
    
    # Don't plot zero values and generate mask to avoid them
    zero_mask = np.zeros_like(matrix)
    zero_mask[np.where(matrix == 0)] = True
    
    #Create two separate masks to plot the diagonal and off diagonal values differently
    diag = np.eye(*matrix.shape, dtype=bool)
    mask = zero_mask + diag
    mask[np.where(mask != 0)] = True
    
    #Set tick label spacing
    xticklabels, yticklabels = tickspace, tickspace
    
    # Create pandas dataframe with row/col labels if available
    if row_labels and col_labels:
        df = pd.DataFrame(matrix, index = row_labels, columns = col_labels)
    elif row_labels:
        df = pd.DataFrame(matrix, index = row_labels)
        yticklabels=False
    elif col_labels:
        df = pd.DataFrame(matrix, columns = col_labels)
        xticklabels=False
    else:
        df = pd.DataFrame(matrix)
        xticklabels, yticklabels = False, False
    
    #Plot heatmap
    if show_values:
        if show_values == True:
            #Plot off diagonal differences with values
            sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                        xticklabels=xticklabels, yticklabels=yticklabels,
                        linewidths=.25, square=True, cbar_kws = dict(pad=-0.03),
                        annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']})
        else:
            #Plot off diagonal differences without values
            sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                        xticklabels=xticklabels, yticklabels=yticklabels,
                        linewidths=.25, square=True, cbar_kws = dict(pad=-0.03)) 
        #Plot diagonal
        sns.heatmap(df, mask = ~diag, cmap = diag_cmap, vmin = 0, vmax = 1, 
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    linewidths=.25, square=True, 
                    annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']})
    else:
        #Plot off diagonal differences
        sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                    linewidths=.25, xticklabels=xticklabels, yticklabels=yticklabels,
                    ax=ax, square=True, cbar_kws = dict(pad=-0.03))
        #Plot diagonal
        sns.heatmap(df, mask = ~diag, cmap = diag_cmap, vmin = 0, vmax = 1, 
                    linewidths=.25, xticklabels=xticklabels, yticklabels=yticklabels,
                    ax=ax, square=True)
    
    # Label colorbars and set axis font size
    cbar_offdiag = ax.collections[0].colorbar
    cbar_offdiag.ax.set_title('diff')
    cbar_offdiag.ax.tick_params(labelsize=fontsizes['small'])
    
    cbar_diag = ax.collections[1].colorbar
    cbar_diag.ax.tick_params(labelsize=fontsizes['small'])
    cbar_diag.ax.locator_params(nbins=11)
    cbar_diag.ax.set_title('opinion')
    
    # Draw frame for heatmap
    linewidth = 5
    ax.axvline(x=0, color='k',linewidth=linewidth)
    ax.axvline(x=matrix.shape[0], color='k',linewidth=linewidth)
    ax.axhline(y=0, color='k',linewidth=linewidth)
    ax.axhline(y=matrix.shape[1], color='k',linewidth=linewidth)
    
    #ax.set_ylabel("Confidence Bound (c)", fontsize = fontsizes['medium'])
    #ax.set_xlabel("Compromise Parameter (m)", fontsize = fontsizes['medium'])
    ax.set_title(title, fontsize = fontsizes['large'], loc = 'left')
    
    #save file if specified
    if savefile:
        plt.savefig(savefile, bbox_inches='tight', facecolor='white')
    plt.show()
    return

def animate_matrix(matrix_list, d, matrix_timestep, T,
                   fps = 2, repeat = False,
                   savefile = False, figsize = (12,10),
                   row_labels = False, col_labels = False, tickspace = 1,
                   show_values = False, decimal_places = 2, 
                   fontsizes = {'small': 10, 'medium': 12, 'large':20}):
    
    """
    Animated heatmaps of visualization matrix of effective subgraph until DW model converges
    
    Parameters
    ------------
    matrix_list : list of numpy.ndarray
        List of adjacency matrices of effective subgraph to visualize/animate
        i.e. list of output['visualization_matrix'] of DW when return_matrix = True
    d : float
        Confidence bound, which sets the max value of colormap for off diagonal entries
        NOTE: THE CONFIDENCE BOUND IS CALLED d HERE, BUT c IN THE PAPER
    matrix_timestep : int
        The timestep between generated visualization matrices.
        This properly sets the titles of the animation matrices.
    T : int
        Final timestep of simulation
        This properly sets the titles of the animation matrices
    fps : int, default = 2
        Frames per second for the animation
    repeat: bool, default = False
        Whether to repeat animation when played in IDE (i.e. in jupyter lab widgets)
    savefile : string, optional
        File name to save the animation to if present. Should be a .mp4 extension
    
    Other Parameters
    ----------------
    figsize : tuple, default = (12,10)
        Tuple with two values giving figure dimensions in inches
    row_labels, col_labels : array_like, optional
        Row labels and column labels for heatmap
    tickspace : int, default = 1
        The spacing/frequency of x and y axis row/col labels if using
    show_values : bool or string, default = False
        If True, print values of each nonzero matrix value in the heatmap
        If 'diagonal' then only print values on the diagonal
    decimal_places : int, default = 2
        Number of decimal places to print for matrix values if show_values = True
    fontsizes : dictionary
        Dictionary with keys 'small', 'medium', 'large' of fontsizes for plot
    
    Returns
    -------
    Shows (and saves if savefile is specified) animation of matrix_list
    """
    
    # Get number of frames = number of matrices
    n_frames = len(matrix_list)
    
    # Create new plot
    fig, ax = plt.subplots(figsize = figsize, facecolor = 'white')
    
    #Set tick label spacing
    xticklabels, yticklabels = tickspace, tickspace
    
    # Define colormaps
    offdiag_cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", ['0.1', '0.99'])
    diag_cmap = 'seismic'
    
    # Calculate millisecond delay interval from fps
    interval = 1000/fps
    
    def generate_plot(matrix, colorbar, title = False):
        """
        Generate seaborn heatmap plot of matrix with boolean colorbar for whether to generate/display a new colorbar
        """
        
        # Create pandas dataframe with row/col labels if available
        if row_labels and col_labels:
            df = pd.DataFrame(matrix, index = row_labels, columns = col_labels)
        elif row_labels:
            df = pd.DataFrame(matrix, index = row_labels)
            yticklabels=False
        elif col_labels:
            df = pd.DataFrame(matrix, columns = col_labels)
            xticklabels=False
        else:
            df = pd.DataFrame(matrix)
            xticklabels, yticklabels = False, False

        # Don't plot zero values and generate mask to avoid them
        zero_mask = np.zeros_like(matrix)
        zero_mask[np.where(matrix == 0)] = True

        #Create two separate masks to plot the diagonal and off diagonal values differently
        diag = np.eye(*matrix.shape, dtype=bool)
        mask = zero_mask + diag
        mask[np.where(mask != 0)] = True

        #Plot heatmap
        if show_values:
            if show_values == True:
                #Plot off diagonal differences with values
                sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                            xticklabels=xticklabels, yticklabels=yticklabels,
                            linewidths=.25, square=True, cbar_kws = dict(pad=-0.03),
                            annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']},
                           cbar = colorbar)
            else:
                #Plot off diagonal differences without values
                sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                            xticklabels=xticklabels, yticklabels=yticklabels,
                            linewidths=.25, square=True, cbar_kws = dict(pad=-0.03),
                           cbar = colorbar) 
            #Plot diagonal
            sns.heatmap(df, mask = ~diag, cmap = diag_cmap, vmin = 0, vmax = 1, 
                        xticklabels=xticklabels, yticklabels=yticklabels,
                        linewidths=.25, square=True, 
                        annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']},
                        cbar = colorbar)
        else:
            #Plot off diagonal differences
            sns.heatmap(df, mask = mask, cmap = offdiag_cmap, vmin = 0, vmax = d, 
                        linewidths=.25, xticklabels=xticklabels, yticklabels=yticklabels,
                        ax=ax, square=True, 
                        cbar = colorbar, cbar_kws = dict(pad=-0.05))
            #Plot diagonal
            sns.heatmap(df, mask = ~diag, cmap = diag_cmap, vmin = 0, vmax = 1, 
                        linewidths=.25, xticklabels=xticklabels, yticklabels=yticklabels,
                        ax=ax, square=True,
                        cbar = colorbar)
        
        # Label colorbars and set axis font size if we have a colorbar
        if colorbar:
            cbar_diag = ax.collections[0].colorbar
            cbar_diag.ax.set_title('diff')
            cbar_diag.ax.tick_params(labelsize=fontsizes['small'])
            cbar_offdiag = ax.collections[1].colorbar
            cbar_offdiag.ax.tick_params(labelsize=fontsizes['small'])
            cbar_offdiag.ax.set_title('opinion')
            #ax.tick_params(axis='both', labelsize=fontsizes['small'])

        # Draw frame for heatmap
        linewidth = 5
        ax.axvline(x=0, color='k',linewidth=linewidth)
        ax.axvline(x=matrix.shape[0], color='k',linewidth=linewidth)
        ax.axhline(y=0, color='k',linewidth=linewidth)
        ax.axhline(y=matrix.shape[1], color='k',linewidth=linewidth)
    
        if title:
            ax.set_title(title, fontsize = fontsizes['large'], loc = 'left')
    
    def init():
        """
        Initialize animation with initial opinion matrix and generate colorbar
        """
        plt.cla()
        matrix = matrix_list[0]
        title = 't = 0'
        generate_plot(matrix, colorbar = True, title = title)
        
    def animate(i):
        """
        Define animation instructions to plot the next matrix for the ith frame
        """
        plt.cla()
        #select the matrix
        matrix = matrix_list[i]
        
        # Calculate t for the title
        if i == len(matrix_list)-1:
            t = T
        else:
            t = i*matrix_timestep 
        title = 't = ' + str(t)
        
        generate_plot(matrix, colorbar = False, title = title)
    
    # Generate and return animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = n_frames, interval = interval, repeat = repeat, cache_frame_data = False, blit=True)
    
    if savefile:
        print('saving file')
        videowriter = animation.FFMpegWriter(fps=fps) 
        anim.save(savefile, writer=videowriter)
        print('done saving file')
    
    return anim

def plot_heatmap(matrix, ds, mus, vmin = None, vmax = None, savefile = None, 
                 show_values = False, decimal_places = 0, tickspace = 1, title = None, 
                 figsize = (10,10), fontsizes = {'small': 12, 'medium': 14, 'large':16}):
    """
    Function to plot head map of DW simulation results from varying confidence bound d and compromise mu 
    
    Parameters
    ------------
    matrix : numpy.ndarray
        Matrix of heatmap values where entry ij represents the value obtained
        from the ith confidence bound (d) and jth compromise parameter (mu)
    ds : list of float
        List of confidence bounds, labels the rows of the matrix
        Note that labels go from top row to bottom row, 
        so the list ds should be in descending order
        NOTE: THE CONFIDENCE BOUND IS CALLED d HERE, BUT c IN THE PAPER
    mus : list of float
        List of compromise parameters, labels the columnss of the matrix
        Note that labels go from left to right, 
        so the list mus should be in ascending order
        NOTE: THE COMPROMISE PARAMETER IS CALLED mu HERE, BUT m IN THE PAPER
    vmin : float, optional
        Sets minimum value for colorbar and coloring scheme if provided. 
    vmax : float, optional
        Sets maximum value for colorbar and coloring scheme if provided. 
    savefile : string, optional
        File name to save the animation to if present. Should be a .mp4 extension
    
    Other Parameters
    ----------------
    show_values : bool, default = False
        If True, show values of each matrix entry
    decimal_places : int, default = 0
        Number of decimal places to print for matrix values if show_values = True
    tickspace : int, default = 1
        The spacing/frequency of x and y axis row/col labels if using
    title : string, optional
        Title to display on plot
    figsize : tuple, default = (10,10)
        Tuple with two values giving figure dimensions (width, height) in inches
    fontsizes : dictionary
        Dictionary with keys 'small', 'medium', 'large' of fontsizes for plot
    
    Returns
    -------
    Shows (and saves if savefile is specified) the plotted heatmap
    """

    #Create plot
    fig, ax = plt.subplots(figsize = figsize)
    
    #Set tick label spacing
    xticklabels, yticklabels = tickspace, tickspace
    
    #Set min and max for colorbar
    # Keep using vmin if given, otherwise take it from matrix values
    if not vmin:
        # If we have -inf values, i.e. if a value is ln(0), then set to 0 so it will plot
        if (np.amin(matrix) == - np.inf):
            matrix[matrix == - np.inf] = 0
            vmin = 0
        else:
            vmin = round_decimals_down(np.nanmin(matrix), decimal_places)
    
    # Keep using vmax if given, otherwise take it from matrix values
    if not vmax:
        vmax = round_decimals_up(np.nanmax(matrix), decimal_places)
    
    #Convert matrix to dataframe for axis labels
    df = pd.DataFrame(matrix, index = ds, columns = mus)
    
    #Plot heat map with or without entry values depending on show_values
    if show_values:
        ax = sns.heatmap(df, cmap = "rainbow", vmin = vmin, vmax = vmax, annot=True, fmt="0."+str(decimal_places)+"f", annot_kws={"size":fontsizes['small']})
    
    else:
        ax = sns.heatmap(df, cmap = "rainbow", vmin = vmin, vmax = vmax)
    
    #Set colorbar and tick label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsizes['small'])
    ax.tick_params(axis='both', labelsize=fontsizes['small'])
    
    #Set axis labels, title and fontsize
    ax.set_ylabel("Confidence Bound (d)", fontsize = fontsizes['medium'])
    ax.set_xlabel("Compromise Parameter (m)", fontsize = fontsizes['medium'])
    ax.set_title(title, fontsize = fontsizes['large'])
    
    #save file if specified
    if savefile:
        plt.savefig(savefile, bbox_inches='tight', facecolor='white')
    plt.show()
    return
    
# ----------------------------------------------------------------------------------------    


#Function to round up
def round_decimals_up(number:float, decimals:int=1):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

#Function to round down
def round_decimals_down(number:float, decimals:int=1):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

#Define function to print average +/1 standard deviation with proper precision
def print_avg_with_std(avg, std):
    #If we have a numpy array, we want to return an array again, but need to fill in entries elementwise
    if isinstance(avg, np.ndarray):
        
        if avg.shape != std.shape:
            raise ValueError('Avg and Std numpy arrays must be of the same shape')
        
        original_shape = avg.shape #save the original array shape
        
        #Flatten everything for convinient indexing, we will reshape at the end
        avg = avg.reshape((avg.size))
        std = std.reshape((std.size))
        results = np.empty(avg.shape, dtype=object)
        
        #Get the number of decimal places to round to
        decimals = np.where(std > 1e-15, int(0) - np.floor(np.log10(std)).astype(int), int(1))
        # decimals = int(1) - np.floor(np.log10(std)).astype(int)
        
        for i in range(avg.size):
            if decimals[i] <= 3:
                truncated_std = round(std[i], decimals[i])
                truncated_avg = round(avg[i], decimals[i])

                # results[i] = str(truncated_avg) + u"\u00B1" + str(truncated_std)
                try:
                    results[i] = ( "{:.{decimals}f}".format(truncated_avg, decimals = decimals[i])
                                   + u"\u00B1" + "{:.{decimals}f}".format(truncated_std, decimals = decimals[i]) )
                except:
                    results[i] = ( "{:.0f}".format(truncated_avg)
                                   + u"\u00B1" + "{:.0f}".format(truncated_std) )
            else:
                avg_exponent = np.floor(np.log10(abs(avg[i]))).astype(int)
                avg_significand = avg[i] * 10**(-avg_exponent)
                std_exponent = np.floor(np.log10(abs(std[i]))).astype(int)
                std_significand = std[i] * 10**(-std_exponent)
                
                if std_exponent < -3:
                    if avg_exponent > -2:
                        results_string = "{:.{}f}".format(avg[i], 3)
                        results_string = results_string + u"\u00B1" + "{:.{}f}".format(std[i], 3)
                    
                    else:
                        avg_significand = avg[i] * 10**(-std_exponent)
                        results_string = "(" + "{:.0f}".format(avg_significand) + u"\u00B1" + "{:.0f}".format(std_significand) + ")"
                        results_string = results_string + r"$\times 10^{" + "{:g}".format(std_exponent) + r"}$"
                else:
                    decimals = abs(std_exponent)
                    results_string = "{:.{}f}".format(avg[i], decimals)
                    results_string = results_string + u"\u00B1" + "{:.{}f}".format(std[i], decimals)
                
                results[i] = results_string
                #results[i] = ("{:.1f}".format(avg_significand) + "e{:g}".format(avg_exponent) + u"\u00B1"
                #              "{:.1f}".format(std_significand) + "e{:g}".format(std_exponent) )
                
                # results[i] = ("{:.1f}".format(avg_significand) + r"$\times 10^{{{:g}}}$".format(avg_exponent) + u"\u00B1"
                #               "{:.1f}".format(std_significand) + r"$\times 10^{{{:g}}}$".format(std_exponent) )
                
                #results[i] = ( "{:.1e}".format(avg[i]) + u"\u00B1" + "{:.1e}".format(std[i]))
                
        results = np.array(results).reshape(original_shape)
        return results
    
    # If we just have scalar values, just return the one print out
    else:
        decimals = int(0) - int(math.floor(math.log10(std)))
        
        if decimals[i] <= 3:
            truncated_std = round(std, decimals)
            truncated_avg = round(avg, decimals)

            return ( "{:.{decimals}f}".format(truncated_avg, decimals = decimals)
                               + u"\u00B1" + "{:.{decimals}f}".format(truncated_std, decimals = decimals) )
        else:
            return "{:.1f}".format(avg) + u"\u00B1" + "{:.1f}".format(std)

        
#Define function to print values to a certain number of decimals with nicer exponential notation
def print_sigfigs_nice(val, sigfigs):
    '''
    
    Parameters
    ----------
    val : float or np.array of float
        Value(s) to convert to nice strings for printing
    decimals : number of significant figures to display
    '''
    
    #If we have a numpy array, we want to return an array again, but need to fill in entries elementwise
    if isinstance(val, np.ndarray):
        
        original_shape = val.shape #save the original array shape
        
        #Flatten everything for convinient indexing, we will reshape at the end
        val = val.reshape((val.size))
        results = np.empty(val.shape, dtype=object)
        
        for i in range(val.size):
            
            if val[i] == 0:
                results[i] = str(0)
        
            else:
                #See what power of 10 the value is and round the to number of sigfigs for the significand
                exponent = np.floor(np.log10(abs(val[i]))).astype(int)
                significand = val[i] * 10**(-float(exponent))
                significand = round(significand, sigfigs - 1)

                # print("value:", val[i], "; significand:", significand, "; exponent:", exponent)

                #For exponents near zero, don't bother with exponential format and just display the number truncated to desired sigfigs
                if exponent >= -1 and exponent <= 5:
                    decimals = (sigfigs - 1) - exponent
                    results_string = str(round(significand, decimals))

                #Otherwise display number in nicer looking exponential notation
                else:
                    results_string = str(significand) + r"$\times 10^{" + "{:g}".format(exponent) + r"}$"
                
                results[i] = results_string
        
        results = np.array(results).reshape(original_shape)
        return results
        
    # If we just have scalar values, just return the one print out
    else:
            
        if val == 0:
            return str(0)
        
        else:
            
            #See what power of 10 the value is and round the to number of sigfigs for the significand
            exponent = np.floor(np.log10(abs(val))).astype(int)
            significand = val * 10**(-float(exponent))
            significand = round(significand, sigfigs - 1)

            #For exponents near zero, don't bother with exponential format and just display the number truncated to desired sigfigs
            if exponent >= -2 and exponent <= 5:
                results_string = "{:.0f}".format( significand * 10**(exponent) ) 

            #Otherwise display number in nicer looking exponential notation
            else:
                results_string = str(significand) + r"$\times 10^{" + "{:g}".format(exponent) + r"}$"

            return results_string