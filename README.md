# Adaptive-Edge-BCM
This repository contains the source code and plots for our bounded-confidence model of opinion dynamics with adaptive edge probabilities. Our paper on this model is currently available on arXiv at https://arxiv.org/abs/2605.20418. This repository was created by Yuexuan (Yolanda) Wu and Leila Thompsky. 

This model is a generalization of the Deffuant--Weisbuch (DW) model where we incorporate heterogeneous and adaptive edge weights between pairs of agents. These edge weights govern the interaction probabilities between the agents and update based on the interactions between agents. Our model aims to encode the idea that people are more likely to communicate with individuals with whom they have previously compromised or had other positive interactions.

# Plots
Plots for each of our experiments are contained in the folder corresponding to the graph type and the subfolders corresponding to the graph size or other graph parameters. For example, the plots for our experiments on k-regular cycle-like graphs with 1000 nodes and common degree k = 10 are available at Degree regular cycle-like graphs/k=10. 

Each collection of plots is contained in a subfolder that specifies either a fixed initial edge-weight parameter z (for example, z=0.1) or a fixed edge-weight increase parameter gamma. The plots in a subfolder with z = 0.1 are all run with the initial edge-weight parameter z = 0.1 and the plots in a subfolder with gamma = 0.1 are all run with the edge-weight increase parameter gamma = 0.1. 

# Code
