# Adaptive-Edge-BCM
This repository contains the source code and plots for our bounded-confidence model of opinion dynamics with adaptive edge probabilities. Our paper on this model is currently available on arXiv at https://arxiv.org/abs/2605.20418. This repository was created by Yuexuan (Yolanda) Wu and Leila Thompsky. 

This model is a generalization of the Deffuant--Weisbuch (DW) model where we incorporate heterogeneous and adaptive edge weights between pairs of agents. These edge weights govern the interaction probabilities between the agents and update based on the interactions between agents. Our model aims to encode the idea that people are more likely to communicate with individuals with whom they have previously compromised or had other positive interactions.

# Plots
Plots for each of our experiments are contained in the folder corresponding to the graph type and the subfolders corresponding to the graph size or other graph parameters. For example, the plots for our experiments on k-regular cycle-like graphs with 1000 nodes and common degree k = 10 are available at Degree regular cycle-like graphs/k=10. 

Each collection of plots is contained in a subfolder that specifies either a fixed initial edge-weight parameter z (for example, z=0.1) or a fixed edge-weight increase parameter gamma. The plots in a subfolder with z = 0.1 are all run with the initial edge-weight parameter z = 0.1 and the plots in a subfolder with gamma = 0.1 are all run with the edge-weight increase parameter gamma = 0.1. 

# Networks

The code to make the synthetic graphs we consider in our paper is contained in `synthetic_graphs_experiment.py`. The real-world networks we consider in our paper are the Netscience graph of coauthorships of network scientists and four of the Facebook100 graphs of Facebook friendships. 

The file for the Netscience graph is contained in the Netscience folder. We use an unweighted version of this network and consider only the largest connected component. We also take the largest connected component of each of the Facebook100 graphs. 

The citations for the Netscience and Facebook100 graphs are below: 

M. E. J. Newman, "Finding community structure in networks using the eigenvectors of matrices", *Phy. Rev. E*, 74(3):
036104, 2006. DOI: 10.1103/PhysRevE.74.036104

A. L. Traud, P. J. Mucha, and M. A. Porter, "Social structure of Facebook networks", *Physica A*, 391(16): 4165-4180, 2012. DOI: 10.1016/j.physa.2011.12.021

# Code

The package versions to implement our code are as follows:
```
python            3.8.10
python-igraph     0.11.4
numpy             1.24.4 
pandas            2.0.3
scipy             1.10.1
matplotlib        3.7.5 
seaborn           0.13.2 
```

The code to run our experiments works as follows: 

Our model is implemented in `EdgeWeightedDW.py`. 

To run an experiment, we use `synthetic_graphs_experiment.py`, `netscience_experiment.py`, and `college_networks_experiment.py` for the synthetic graphs, Netscience network, and Facebook100 college networks respectively. The synthetic graphs we consider are complete graphs, Erdos-Renyi graphs, and degree-regular cycle-like graphs. 

Once simulations have been completed, we use `Matfile_Consolidator.py` to calculate our quantities of interest and generate a `.csv` file of the results for each parameter pair $(\gamma, \delta)$. 

To generate the plots presented in our paper, we use `linePlots_fixz_4plot.py` and `linePlots_fixgamma_4plot.py`, which plot the results in our `.csv` files.  `linePlots_fixz_4plot.py` and `linePlots_fixgamma_4plot.py` generate plots for either two values of $\delta$ and two values of $\gamma$, or for two values of $\delta$ and two values of $z_0$, respectively. We also include the files `linePlotsDW_fixz_9plots.py` and `linePlotsDW_fixgamma_9plots.py`,  which generate plots for three values of each parameter in the parameter pairs $(\gamma,\delta)$ or $(z_0, \delta)$. 


