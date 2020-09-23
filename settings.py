#!/usr/bin/env python
# coding: utf-8
## Settings ##

import networkx as nx
from matplotlib import pyplot as plt
import math
import numpy as np
import random
import copy
import statistics as stat

######################################################################################
# Parameterization
######################################################################################

from params import *

def init():

        
    # system parameters
    global max_capacity,max_num_path,min_capacity
    global k_max,alpha,beta,f_min
    global p_in,p_out,Fid_mean,Fid_std,Fid_th
    global G
    global num_request,request,weight,average_dis_request

    # configuration parameters
    global fig_on


    f_min = max(np.floor(max_capacity/max_num_path),1)
    min_capacity = []
    
    if Graph_type == 'square':
    	G = nx.grid_2d_graph(Graph_width,Graph_width)   # m x n square lattice
    if Graph_type == 'hexogonal':
    	G = nx.hexagonal_lattice_graph(Graph_width,Graph_width) # m x n hexagonal lattice
    if Graph_type == 'triangular':
    	G = nx.triangular_lattice_graph(Graph_width,Graph_width)        # m x n triangular lattice
    

    if random_request == 1:
        request = []
        for i in range(num_request):
            temp_request = [(random.randint(0,Graph_width-1),random.randint(0,Graph_width-1)),(random.randint(0,Graph_width-1),random.randint(0,Graph_width-1))]
            request.append(temp_request)
            
    dis_request = []
    for k in request:
        dis_request.append(abs(k[0][0] - k[1][0]) + abs(k[0][1] - k[1][1]))
        
    average_dis_request = sum(dis_request)/num_request
    
    for k in request:
        if k[0] == k[1]:
            print('Warning! Loop request: the same sender and receiver.')

    if len(request) != len(weight):
        print('Warning! Length of request and weight not match.')

    # unpredetermined parameters 
    global pos,all_path_pool,all_real_flow
    global weighted_flow_sum,weighted_flow_min,ave_path_length,ave_var_flow,average_capacity_utilization,variance_capacity_utilization

    pos = []
    all_path_pool = []
    all_real_flow = []
    weighted_flow_sum = 0
    weighted_flow_min = 0
    ave_path_length = 0
    ave_var_flow = 0
    average_capacity_utilization = 0
    variance_capacity_utilization = 0

    nx.set_edge_attributes(G, [], 'request_ID_on_edge')
    nx.set_edge_attributes(G, [], 'flow_on_edge')
    nx.set_edge_attributes(G, [], 'traffic_on_edge')

