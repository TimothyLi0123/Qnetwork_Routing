##Input File##
##Edit paramters before each run.##
##Don't forget to save after update.##

import random

# system parameters 
max_capacity = 100          
max_num_path = 10          # maximum number of path that can take place on one edge


k_max = 10
alpha = 1
beta = 1


p_in = 0.9
p_out = 0.9
Fid_mean = 0.8
Fid_std = 0.1
Fid_th = 0.8


Graph_width = 8
Graph_type = 'square'      # 'square'/'hexogonal'/'triangular'
Graph_type = 'triangular'

num_request = 10

#request = [[(0,0),(2,3)],[(0,1),(3,3)],[(1,2),(3,1)],[(1,2),(5,4)]]    
#request = [[(3,3),(6,6)],[(6,3),(3,6)]]   # -- base case
                           # Requests from the same sender & receiver seen as different requests.                       
request = []
while len(request) < num_request:
    request_temp = [(random.randint(0, Graph_width-1),random.randint(0, Graph_width-1)),(random.randint(0, Graph_width-1),random.randint(0, Graph_width-1))]
    if request_temp not in request and request_temp[0] != request_temp[1]:
        request.append(request_temp)
    
weight = [1]*num_request         # weight of each request

random_request = 0


# configuration parameters

Step1_output = 1           # 0: intermediate output off; 1: intermediate output on
Step3_output = 0 
Step4_output = 1 
Step5_output = 0 
 
fig_on = 0                 # 0: traffic plot off; 1: traffic plot on

failure = 0                # 0: failure off; 1: edge failure; 2: node failure
failure_num = 1            # Number of failed nodes or edges
removed_edges = []

load_existing_lattice = 0  # 0: make new lattice; 1: load existing lattice
save_new_lattice = 1       # 0: save this lattice; 1: drop this lattice
save_run = 1               # 0: save this run; 1: drop this run

save_run_name = 'Run#00111'

load_lattice_name = 'Lattice--Run#0001'


