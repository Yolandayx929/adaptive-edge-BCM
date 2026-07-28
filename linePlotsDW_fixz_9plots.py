'''
Modified code by Leila Thompsky based on code by Grace Li
Date: 2026 

This script generates plots of the results of the simulations for our adaptive edge-weighted DW BCM.
In the paper, we examine the numbers of major and minor clusters, the Shannon entropy, and convergence time. 

For each quantity of interest, this script generates a single figure where each plot shows the quantity of interest versus the confidence bound c. Each plot represents a single edge weight increase and
edge weight decrease parameter pair (gamma, delta) for a fixed initial edge weight z0 (called z in the code). Each plot has different colored curves corresponding to different mu values. We generate each point on the plot from 10 or 20 numerical simulations each with different sets of initial 
opinions drawn uniformly at random. We plot the mean with error range representing one standard deviation across the numerical simulations for that point. 

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

# Import our own adaptive-confidence bounded-confidence module
import sys
sys.path.append('..') #look one directory above
import DW

sns.set_style("ticks",{'axes.grid' : True})

#Change experiment parameters here -----------------------------------------------

#name of experiment folder
graph_name = "Caltech" #If we have a specific graph
network_dataset = False #Set equal to true if we want to use graph name (for the college networks)

graph_type = "degree-regular" #If we are generating a synthetic network with n nodes
n = 1000

#se for standard error, sd for standard deviation
error_type = "sd" 

if graph_type == "degree-regular":
    k = 10

if graph_type == "erdos-renyi":
    p = 0.01

runtoTmax = False
toltest = False
tol = 0.01 #specify the tolerance value

#Specify which quantities to plot
keys = [
        'log(T)', 'entropy',
        'n_major', 'n_minor',
       ]

# edge weight increase parameter gamma and edge weight decrease parameter delta 
gammas = [0.1, 0.5, 0.9] 
deltas = [0.1, 0.5, 0.9] 

gamma_delta_pairs = [(0.0, 1.0)]
for gamma in gammas:
    for delta in deltas:
        gamma_delta_pairs.append((gamma, delta))
print(gamma_delta_pairs)

# Confidence bound and initial edge weights
cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
zs = [0.1] #specify only one value of z 
xtick_list = [0.1, 0.3, 0.5, 0.7, 0.9]

# #Compromise parameter 
mus = [0.1, 0.3, 0.5]

#savefolder name for experiment
if network_dataset:
    experiment = 'college-networks/' + graph_name + ("/runtoTmax" * runtoTmax)
if graph_type == "erdos-renyi":
    experiment = graph_type + str(n) + ("/runtoTmax" * runtoTmax)
else:
    experiment = graph_type + str(n) + ("/runtoTmax" * runtoTmax)

#Plot parameters  ---------------------------------------------------------------------------
#Plot font size parameters
# fontsizes = {'XS': 10, 'S': 15, 'M': 24, 'L':30, 'XL': 35, 'space': 20}
fontsizes = {'XS': 13, 'S': 16, 'M': 24, 'L':25, 'XL': 32, 'XXL':45, 'space':35, 'left_space': 75} 


colorlist = ['#0173b2', '#de8f05', '#029e73']
markerlist = ["o", "s", "^"]

plot_height = 5
plot_width = 5

rows = len(deltas)
cols = len(gammas)
figsize = (plot_width*(cols + 1), plot_height*rows)

bbox_to_anchor = (0.1, 0.85)
baseline_plot_position = 1
    
# bbox_to_anchor = (0.07, 0.49)
# baseline_plot_position = 0
    
plot_name = {
                'log(T)': 'T',
                'n_clusters': 'clusters', 
                'n_minor': 'n_minor',
                'n_major': 'n_major',
                'entropy' : 'entropy',
                'avg_opinion_diff': 'op_diff', 
                'avg_local_agreement' : 'agreement', 
                'avg_local_receptiveness': 'receptiveness',
                'op_mean': 'op_mean', 'op_var': 'op_var', 
                'op_skew': 'op_skew', 'op_kurtosis': 'op_kurtosis'
            }
plot_ylabel = {
                'log(T)': r"$\log_{10}(T_f)$", 
                'n_clusters': 'Number of clusters',
                'n_minor': 'Number of minor clusters',
                'n_major': 'Number of major clusters',
                'entropy': 'Shannon entropy ' + r"$(H(T_f))$",
                'avg_opinion_diff': 'Mean opinion difference',
                'avg_local_agreement': 'Mean local agreement', 
                'avg_local_receptiveness': 'Mean local receptiveness', 
                'op_mean': 'Opinion mean', 'op_var': 'Opinion variance', 
                'op_skew': 'Opinion skew', 'op_kurtosis': 'Opinion kurtosis'
             }

plot_title = plot_ylabel.copy()

#Names for the columns of the consolidated matfiles
columns = ['delta', 'gamma', 'c', 'z', 'opinion_set',
           'n_clusters', 'n_major', 'n_minor',
           'max_diameter', 'T', 'bailout', 
           'avg_opinion_diff', 'avg_local_receptiveness', 'entropy',
           'op_mean', 'op_var', 'op_skew', 'op_kurtosis'
          ]

idx_to_letter = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K'}


# Plot results for each gamma/delta pair ---------------------------------------------------

#Loop through the quantities of interest
for key in keys:
    
    #Get min and max for the vertical axis
    min_y, max_y = 999999999999, 0
    for i in range(len(gamma_delta_pairs)):
        pair = gamma_delta_pairs[i]
        gamma, delta = pair[0], pair[1]

        #Load the simulation results stored csv files (from MatfileConsolidator.py)
        if graph_type == "degree-regular":
            # filename = experiment + '/combined_results/k' + str(k) + '/continue/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
            filename = experiment + "/combined_results" + '/k' + str(k) + '/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
            
        elif graph_type == "erdos-renyi":
            filename = experiment + '/combined_results/p' + str(p) + '/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
            
        else: 
            filename = experiment + '/combined_results/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
        
        sim_results = pd.read_csv(filename)

        sim_results = sim_results[sim_results['mu'].isin(mus)]
        sim_results = sim_results[sim_results['z'].isin(zs)]

        #Make sure the opinion_set, and time steps are integers instead of floats
        sim_results = sim_results.astype({'opinion_set': int, 'T': int})
        #'T_changed': int
        
        #Calculate the log of time steps to make it easier to visualize
        sim_results['log(T)'] = np.log10(sim_results['T'])

        #Update max an min values
        min_y = min(min_y, sim_results[key].min())
        max_y = max(max_y, sim_results[key].max())

    #Create the plot
    fig, axs = plt.subplots(rows, cols + 1, figsize=figsize)
    
    label_idx = 0

    #Generate each subplot
    for row in range(rows):
        for col in range(cols):
            
            label_idx += 1

            ax = axs[row, col+1]
            gamma, delta = gammas[col], deltas[row]

            #Load the simulation results stored csv files (from MatfileConsolidator.py)
            if graph_type == "degree-regular":
                # filename = experiment + '/combined_results/k' + str(k) + '/continue/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
                filename = experiment + "/combined_results" + '/k' + str(k) + '/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
                
            elif graph_type == "erdos-renyi":
                filename = experiment + '/combined_results/p' + str(p) + '/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
                
            else: 
                filename = experiment + '/combined_results/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
                
            sim_results = pd.read_csv(filename)

            sim_results = sim_results[sim_results['mu'].isin(mus)]
            sim_results = sim_results[sim_results['z'].isin(zs)]
            
            #Make sure the opinion_set, and time steps are integers instead of floats
            sim_results = sim_results.astype({'opinion_set': int, 'T': int})
            #'T_changed': int

            #Calculate the log of time steps to make it easier to visualize
            sim_results['log(T)'] = np.log10(sim_results['T'])
            #sim_results['log(T_changed)'] = np.log10(sim_results['T_changed'])
            
            #Convert delta to a string so the legend displays properly
            sim_results['delta'] = str(delta)
            if delta == 1.0:
                sim_results['delta'] = '1'

            #Generate pointplot for visualization for this delta value
            # sns.pointplot(x='c', y=key, ax = ax, data=sim_results)
            sns.lineplot(x ='c', y = key, ax = ax, data = sim_results, errorbar=error_type,
                             hue = "mu",  palette = colorlist, 
                             style = "mu", markers = markerlist, dashes = False, markersize=10) 
            legend = ax.get_legend()
            legend.set_title(r'$\mu$')

            if gamma == 0.0 and delta == 1.0:
                title = "Baseline DW model\n" + r"($\gamma = 0$, $\delta = 1$)"
            else:            
                title = r"$\gamma = $" + str(gamma) +  r"$, \delta = $" + str(delta)
            ax.set_title(title, fontsize = fontsizes['L'], pad = fontsizes['XS'])

            ax.set(ylabel=None)          
            if row == 2:
                ax.set_xlabel(" ", fontsize = fontsizes['space'])
                ax.xaxis.label.set_color('white')
            else:
                ax.set(xlabel=None)
                
            ax.set_xticks(xtick_list)
            ax.tick_params(axis = 'both', labelsize = fontsizes['S'])
            
            #y ticks
            if key in ['frac_edges', 'wt_avg_frac_edges_cluster', 'deleted_within_of_total', 'deleted_within_of_deleted', 'avg_local_receptiveness']:
                ax.set_ylim(-0.05, 1.05)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            elif key == "entropy":
                ax.set_ylim(-0.1, DW.round_decimals_up(max_y, 1))
            elif key == "n_major":
                if network_dataset and graph_name == "netscience":
                    ax.set_ylim(0, 25)
                else:
                    ax.set_ylim(0, DW.round_decimals_up(max_y, 1))
            elif key in ["n_minor", 'n_clusters']:
                max_value = DW.round_decimals_up(max_y, 1)
                if max_value < 10:
                    ax.set_ylim(-0.1, max_value)
                elif max_value < 100:
                    ax.set_ylim(-1, max_value)
                else:
                    ax.set_ylim(-5, max_value) 
            elif key == 'op_var':
                ax.set_ylim(-0.02, DW.round_decimals_up(max_y, 1))
            else:
                ax.set_ylim(DW.round_decimals_down(min_y, 1), DW.round_decimals_up(max_y, 1))
            
            
            ax.get_legend().remove()  

            #For the first gamma plot we generate, create a global legend in the last corner
            if row == 0 and col == 0:
                legend = fig.legend(title = r"$\mu$", loc='center left', bbox_to_anchor=bbox_to_anchor,
                          fontsize = fontsizes['L'], title_fontsize = fontsizes['XL'], markerscale=2)
                
            ax.text(-0.08, 1.2, idx_to_letter[label_idx], transform=ax.transAxes, fontsize=fontsizes['XL'], fontweight='bold', va='top', ha='right')
                
    #Generate the baseline model plot separately
    for i in range(rows):
        if i != baseline_plot_position:
            axs[i, 0].set_axis_off()

    ax = axs[baseline_plot_position, 0]
    gamma, delta = 0.0, 1.0

     #Load the simulation results stored csv files (from MatfileConsolidator.py)
    if graph_type == "degree-regular":
        filename = experiment + '/combined_results/k' + str(k) + '/continue/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
        # filename = experiment + "/combined_results" + '/k' + str(k) + '/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
    elif graph_type == "erdos-renyi":
        filename = experiment + '/combined_results/p' + str(p) + '/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'
    else: 
        filename = experiment + '/combined_results/delta' + (str(int(delta)) if delta == 1.0 else str(delta)) + '-gamma' + (str(int(gamma)) if gamma == 0.0 else str(gamma)) + '.csv'

    #indent level
    sim_results = pd.read_csv(filename)
    sim_results = sim_results[sim_results['z'].isin(zs)]

    #baseline file 
    if graph_type == "degree-regular":
        baseline_filename = f"{experiment}/combined_results/k{k}/delta1-gamma0.csv"
        #baseline_filename = f"{experiment}/combined_results/k{k}/continue/delta1-gamma0.csv"
    elif graph_type == "erdos-renyi":
        baseline_filename = f"{experiment}/combined_results/p{p}/delta1-gamma0.csv"
    else:
        baseline_filename = experiment + '/combined_results/delta1-gamma0.csv'
    baseline_results = pd.read_csv(baseline_filename)

    # Append baseline results to sim_results
    sim_results = pd.concat([sim_results, baseline_results], ignore_index=True)

    sim_results = sim_results[sim_results['mu'].isin(mus)]
    
    #Make sure the opinion_set, and time steps are integers instead of floats
    sim_results = sim_results.astype({'opinion_set': int, 'T': int})
    #'T_changed': int

    #Calculate the log of time steps to make it easier to visualize
    sim_results['log(T)'] = np.log10(sim_results['T'])
    #sim_results['log(T_changed)'] = np.log10(sim_results['T_changed'])

    #Generate pointplot for visualization for this delta value
    # sns.pointplot(x='c', y=key, ax = ax, data=sim_results)
    sns.lineplot(x ='c', y = key, ax = ax, data = sim_results, errorbar=error_type,
                     hue = "mu",  palette = colorlist,
                     style = "mu", markers = markerlist, dashes = False, markersize=10)

    legend = ax.get_legend()
    legend.set_title(r'$\mu$')
    title = "Baseline DW model\n" + r"($\gamma = 0$, $\delta = 1$)"   
    ax.set_title(title, fontsize = fontsizes['L'], pad = fontsizes['XS'])

    ax.set(xlabel=None)
    ax.set(ylabel=None)

    ax.set_xticks(xtick_list)
    ax.tick_params(axis = 'both', labelsize = fontsizes['S'])

    #y ticks
    if key in ['frac_edges', 'wt_avg_frac_edges_cluster', 'deleted_within_of_total', 'deleted_within_of_deleted', 'avg_local_receptiveness']:
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
    elif key == "entropy":
        ax.set_ylim(-0.1, DW.round_decimals_up(max_y, 1))
    elif key == "n_major":
        ax.set_ylim(0, DW.round_decimals_up(max_y, 1))
    elif key in ["n_minor", 'n_clusters']:
        max_value = DW.round_decimals_up(max_y, 1)
        if max_value < 10:
            ax.set_ylim(-0.1, max_value)
        elif max_value < 100:
            ax.set_ylim(-1, max_value)
        else:
            ax.set_ylim(-5, max_value)
    elif key == 'op_var':
        ax.set_ylim(-0.02, DW.round_decimals_up(max_y, 1))
    else:
        ax.set_ylim(DW.round_decimals_down(min_y, 1), DW.round_decimals_up(max_y, 1))

    ax.get_legend().remove()
    
    ax.text(-0.08, 1.2, idx_to_letter[0], transform=ax.transAxes, fontsize=fontsizes['XL'], fontweight='bold', va='top', ha='right')
    
    if baseline_plot_position == 1:
        ax.set_ylabel(plot_ylabel[key], fontsize = fontsizes['XXL'], va='center', labelpad = fontsizes["left_space"])
    else:    
        fig.supylabel(plot_ylabel[key], fontsize = fontsizes['XXL'], va='center')
        
    fig.supxlabel(" "*6 + r"Confidence bound ($c$)", fontsize = fontsizes['XXL'])
    
    #Save the plot
    if graph_type == "degree-regular":
        #savefile = f"{experiment}/combined_plots/k{k}/continue/big_grid-z.1/"
        savefile = f"{experiment}/combined_plots/k{k}/big_grid-z.1/"
    elif graph_type == "erdos-renyi":
        savefile = f"{experiment}/combined_plots/p{p}/big_grid-z.1/"
    else:
        savefile = experiment + "/combined_plots/big_grid-z.1/"
    if error_type == "se":
        savefile = savefile + "with_se/"
    if not os.path.exists(savefile):
        os.makedirs(savefile)
    if network_dataset:
        savefile = savefile + f"{graph_name}-grid--"
    else:
        savefile = savefile + f"{graph_type}{n}-grid--"
    savefile = savefile + plot_name[key] + f".png"
    
    fig.tight_layout(w_pad=4, h_pad=3)
    plt.savefig(savefile, bbox_inches='tight', facecolor='white')
    plt.show()