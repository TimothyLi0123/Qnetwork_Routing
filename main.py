#!/usr/bin/env python
# coding: utf-8
## Main workflow ##

def Step1_Capacity_Initialization(Step1_output):
    
    global G,max_capacity,min_capacity,f_min,p_in,p_out,Fid_mean,Fid_std,Fid_th,pos
    
    original_edge_num = len(G.edges())
    
    
    capacity = []     ## exogeneous capacity 
    nx.set_edge_attributes(G, capacity, 'capacity')

    count = 0
    for k in list(G.edges):
        if count < len(capacity):
            G.edges[k]['capacity'] = capacity[count]
        else:
            G.edges[k]['capacity'] = max_capacity
        count = count + 1

    ## Fidelity --> Capacity

    nx.set_edge_attributes(G,[], 'Fidelity')

    min_capacity = max_capacity

    for k in list(G.edges):
        edgek_fid = np.random.normal(Fid_mean, Fid_std)
        G.edges[k]['Fidelity'] = edgek_fid

        if edgek_fid <= 0.5: 
            G.edges[k]['capacity'] = 0
            G.remove_edge(k[0],k[1])      
            continue

        if 0.5 < edgek_fid < Fid_th:
            
            G.edges[k]['capacity'] = G.edges[k]['capacity']//PurificationNum(edgek_fid, Fid_th)    
            G.edges[k]['capacity'] = np.random.binomial(G.edges[k]['capacity'], p_out)
            
        if edgek_fid >= Fid_th:
            G.edges[k]['capacity'] = np.random.binomial(G.edges[k]['capacity'], p_out)

        if G.edges[k]['capacity'] < max_num_path:
            G.remove_edge(k[0],k[1])
            continue

        if min_capacity > G.edges[k]['capacity']:      # update min capacity
            min_capacity = G.edges[k]['capacity']

    f_min = max(np.floor(min_capacity/max_num_path),1)
    
    # Grid pos
    pos = nx.spectral_layout(G)
    for di in G.nodes():
        pos[di] = [di[0],di[1]]
      
    # plot grid   
    if Step1_output == 1:
        plt.figure(figsize=(6,6)) 
        nx.draw(G, pos, edge_color='k', with_labels=False, font_weight='light', node_size= 40, width= 1)

        plt.show()
        
def Step2_Path_Determination():
    
    global G,request,num_request,all_path_pool,k_max
    
    all_path_pool = []
    count_request = 0

    for r in request:                             # each request
        count_request = count_request + 1 
        temp_request_s = r[0]
        temp_request_t = r[1]

        if temp_request_s not in G.nodes() or temp_request_t not in G.nodes():  ## Failure: Request out of grid
            path_pool = []
            path_length = math.inf
            all_path_pool.append([path_pool,[path_length]*len(path_pool)])
            print('Warning! Request out of grid:' + str(r))
            continue

        if not nx.has_path(G, temp_request_s, temp_request_t):                  ## Failure: No path in between
            path_pool = []
            path_length = math.inf
            all_path_pool.append([path_pool,[path_length]*len(path_pool)])
            print('Warning! No Path exist for request:' + str(r))
            continue
        
        path_pool = k_shortest_paths(G, temp_request_s, temp_request_t, k_max, weight=None)

        count_path = 0

        for l in path_pool:                    # each path for this request
            count_path =  count_path + 1

            for ll in range(len(l)-1):         # each edge in this path

                temp = list(G.edges[(l[ll],l[ll+1])]['request_ID_on_edge'])
                temp.append([count_request,count_path,len(l)-1,ll]) # append(request,path,path_length,order in the path)  
                G.edges[(l[ll],l[ll+1])]['request_ID_on_edge'] = temp

        all_path_pool.append([path_pool,[len(pp)-1 for pp in path_pool]])
        
def Step3_Capacity_Allocation_PF(Step3_output):
    
    global G,all_path_pool,max_num_path,num_request,f_min

    ## P1 -- Initialize flow vector for each path --- and path index

    flow_on_path = []
    all_path_ind = []
    
    for i in range(len(all_path_pool)):                  # for each request

        flow_on_path.append([])

        for j in range(len(all_path_pool[i][0])):       # for each path
         
            flow_on_path[i].append(0)    # initial 0 
            all_path_ind.append([i,j])    
    
    if all_path_ind == []:
        return
    
    ## P2 -- Progressive Filling
    
    un_sat_edge = list(G.edges())
    un_sat_path = copy.deepcopy(all_path_ind)
    
    t_pf = 0
    while True: 
        
        t_pf = t_pf + 1
        ## attempt made ##
        
        outbound_flag = 0
        
        attempted_flow = copy.deepcopy(flow_on_path)
        
        for i in range(len(un_sat_path)):
            
            ind1 = un_sat_path[i][0]  
            ind2 = un_sat_path[i][1]
            
            attempted_flow[ind1][ind2] = attempted_flow[ind1][ind2] + 1
        
        ## check if outbound ##
        
        sat_edge = []
            
        for k in list(G.edges()):
            
            attempted_flow_on_edge = 0
        
            for i in range(len(G.edges[k]['request_ID_on_edge'])):
            
                ind1 = G.edges[k]['request_ID_on_edge'][i][0]-1
                ind2 = G.edges[k]['request_ID_on_edge'][i][1]-1
            
                attempted_flow_on_edge = attempted_flow_on_edge + attempted_flow[ind1][ind2]

            if attempted_flow_on_edge > G.edges[k]['capacity']:
                print('Warning! Outbound on edge' + str(k))
                outbound_flag = 1
                break
            
            if t_pf == 1:
                sat_path = []
                
            if attempted_flow_on_edge - G.edges[k]['capacity'] > - len([item for item in [kk[0:2] for kk in G.edges[k]['request_ID_on_edge']] if item not in sat_path]):
                sat_edge.append(k)
            
        if outbound_flag == 1:
            break
             
        ## attempt succeeded -- no outbound ##
        
        flow_on_path = copy.deepcopy(attempted_flow)
        
        for i in range(len(sat_edge)):
            
            for j in range(len(G.edges[sat_edge[i]]['request_ID_on_edge'])):  
        
                temp_sat_path = G.edges[sat_edge[i]]['request_ID_on_edge'][j][0:2]
                if temp_sat_path not in sat_path:
                    sat_path.append(temp_sat_path)
        
        un_sat_edge = [item for item in list(G.edges()) if item not in sat_edge]   #list(set(list(G.edges())) - set(sat_edge))   
        un_sat_path = [item for item in list(all_path_ind) if item not in sat_path]#list(set(list(all_path_ind)) - set(sat_path))
        
    final_flow_on_path = copy.deepcopy(flow_on_path) 
    
    # Allocation complete  

    for k in list(G.edges()):
        
        flow_on_edge_raw = []
        
        for i in range(len(G.edges[k]['request_ID_on_edge'])):
            
            ind1 = G.edges[k]['request_ID_on_edge'][i][0]-1
            ind2 = G.edges[k]['request_ID_on_edge'][i][1]-1
            
            flow_on_edge_raw.append(final_flow_on_path[ind1][ind2])

        G.edges[k]['flow_on_edge'] = flow_on_edge_raw

        if sum(flow_on_edge_raw) > G.edges[k]['capacity']:
            print('Warning! Edge' + str(k) + ' exceeds capacity.')

        if Step3_output == 1:
            print('Edge'+ str(k) + '---' + str(flow_on_edge_raw))

def Step4_Routing_Performance(Step4_output):

    global G,num_request,all_path_pool,max_capacity,all_real_flow,weighted_flow_sum,weighted_flow_min,ave_path_length,ave_var_flow,p_in,Jain_request,Jain_path
    
    flow_prob = []            # unweighted flow rate, with p_in incorporated

    all_real_flow = []        # record all realized flow

    for i in range(num_request):                      # for each request

        flow_path = []
        num_path = len(all_path_pool[i][0])

        for j in range(num_path):                     # for each path

            list_edge = all_path_pool[i][0][j]

            min_flow = max_capacity
            
            flag_path_cancellation = 0
            
            for k in range(len(list_edge)-1):           # for each edge

                # find the index

                if [i+1,j+1] not in  [w[:2] for w in G.edges[list_edge[k],list_edge[k+1]]['request_ID_on_edge']]:
                    flag_path_cancellation = 1
                    break

                indd = [w[:2] for w in G.edges[list_edge[k],list_edge[k+1]]['request_ID_on_edge']].index([i+1,j+1]) 

                if indd > len(G.edges[list_edge[k],list_edge[k+1]]['flow_on_edge']):
                    print('Warning! Index does not match on edge: '+ str(list_edge[k]) +','+ str(list_edge[k+1]))

                if G.edges[list_edge[k],list_edge[k+1]]['flow_on_edge'] == []:
                    print('Warning! Flow allocation incomplete on edge: '+ str(list_edge[k]) +','+ str(list_edge[k+1]))
                
                temp = G.edges[list_edge[k],list_edge[k+1]]['flow_on_edge'][indd]

                if temp < min_flow:
                    min_flow = temp

            if min_flow == max_capacity and flag_path_cancellation != 1:
                print('Warning! Min flow = Max flow on request '+ str(i+1) + ', path ' + str(j+1) + ', check if error.')

            if flag_path_cancellation == 1:
                min_flow = 0
                
            flow_path.append(int(min_flow))

        path_length = all_path_pool[i][1]

        if Step4_output == 1:
            print('Request #' + str(i+1) + '----'+ str(request[i][0]) + ' to ' + str(request[i][1]))
            print('Weight:' + str(weight[i]))
            print('Number of paths:' + str(num_path))
            print('Flow of each path:' + str(flow_path))
            print('Length of each path:' + str(path_length))
            print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
            
        flow_prob.append(np.dot(flow_path,list(np.power(p_in, [x-1 for x in path_length]))))    

        all_real_flow.append([flow_path,path_length])


    # objective function #1: 
    weighted_flow_sum = np.dot(weight,flow_prob)              # weighted sum flow 

    # objective function #2:
    weighted_flow_min = min(np.multiply(weight,flow_prob))    # weighted min flow 

    # objective function #3: average path length (normalized by shortest path length)
    
    ave_path_length_request = []
    
    for i in range(num_request):                      # for each request
        if all_path_pool[i][0] != []:
            temp = np.dot(all_real_flow[i][0],all_real_flow[i][1])/sum(all_real_flow[i][0])/nx.shortest_path_length(G, source=request[i][0], target=request[i][1])
            ave_path_length_request.append(temp)
        else:
            ave_path_length_request.append(None)
    
    if len(list(filter(None,ave_path_length_request)))!= 0:
        ave_path_length = sum(filter(None,ave_path_length_request))/len(list(filter(None,ave_path_length_request)))
    else:
        ave_path_length = 1234567
        
    # objective function #4: average flow variance 
    
    var_flow_request = []
    for i in range(num_request):                      # for each request
        if all_path_pool[i][0] != []:
            temp = np.var(list(np.divide(all_real_flow[i][0],[sum(all_real_flow[i][0])]*len(all_real_flow[i][0]))))
            var_flow_request.append(temp)
        else:
            var_flow_request.append(None)
    
    if len(list(filter(None,var_flow_request))) != 0:
        ave_var_flow = sum(filter(None,var_flow_request))/len(list(filter(None,var_flow_request)))
    else:
        if var_flow_request == [0]*len(var_flow_request):
            ave_var_flow = 0
        else:
            ave_var_flow = 1234567
            
    # objective function #5: fairness over requests
    
    flow_request = []
    for i in range(num_request):                      # for each request
        if all_path_pool[i][0] != []:
            temp = sum(all_real_flow[i][0])
            flow_request.append(temp)
        else:
            flow_request.append(None)
            
    flow_request = list(filter(None,flow_request))    
    if sum(np.square(flow_request)) != 0:
        Jain_request = sum(flow_request)**2/len(flow_request)/sum(np.square(flow_request))
    else:
        Jain_request = 0
    
    if flow_request != []:
        if max(flow_request) != 0:
            Gs_request = np.prod([math.sin(k*math.pi/2/max(flow_request)) for k in flow_request])          
        else:
            Gs_request = 0
        
    # objective function #6: fairness over paths
    
    flow_path = []
    for i in range(num_request):                      # for each request
        if all_path_pool[i][0] != []:
            temp = all_real_flow[i][0]
            flow_path = flow_path + temp
    
    # flow_path = list(filter((0).__ne__,flow_path))  # Activate only when using Gs_path
    
    if sum(np.square(flow_path))!=0:
        Jain_path = sum(flow_path)**2/len(flow_path)/sum(np.square(flow_path))
    else: 
        Jain_path = 0
    if flow_path != []: 
        if max(flow_path) !=0:
            Gs_path = np.prod([math.sin(k*math.pi/2/max(flow_path)) for k in flow_path])          
        else:
            Gs_path = 0
        
def Step5_Capacity_Utilization(Step5_output):
    
    global G,all_real_flow,average_capacity_utilization,variance_capacity_utilization

    sum_capacity = 0
    count_used_edge = 0

    for k in list(G.edges):

        edge_capacity = G.edges[k]['capacity']
        edge_request = G.edges[k]['request_ID_on_edge']
        if edge_request == []:
            continue

        real_capacity = 0
        for i in range(len(edge_request)):

            ind1 = G.edges[k]['request_ID_on_edge'][i][0]-1    # which request
            ind2 = G.edges[k]['request_ID_on_edge'][i][1]-1    # which path
            ind3 = G.edges[k]['request_ID_on_edge'][i][2]      # path length

            if all_real_flow[ind1][1][ind2] != ind3:           # Check: path length does not match
                print('Warning! Path length does not match.')
                break

            real_capacity = real_capacity + all_real_flow[ind1][0][ind2]
        
        if Step5_output == 1:
            print('Edge--' + str(k) + '--' + str(round(real_capacity/edge_capacity*100,2)) + '%') 

        if real_capacity/edge_capacity > 1:                    # Check: capacity overflow
                print('Warning! Capacity Utilization > 1.')

        G.edges[k]['traffic_on_edge'] = round(real_capacity/edge_capacity*100,3)

        count_used_edge = count_used_edge + 1
        sum_capacity = sum_capacity + real_capacity/edge_capacity    
    if count_used_edge != 0:
        average_capacity_utilization = sum_capacity/count_used_edge 
    else:
        average_capacity_utilization = 0
        
    list_utilization = [G.edges[k]['traffic_on_edge'] for k in G.edges()]
    list_utilization = list(filter(None, list_utilization))
    
    if len(list_utilization)>1:
        variance_capacity_utilization = stat.variance(list_utilization)/10000
    else:
        variance_capacity_utilization = 0
    
def Step6_Results():
    
    global G,average_capacity_utilization,variance_capacity_utilization,weighted_flow_sum,weighted_flow_min,ave_path_length,ave_var_flow,pos,Step3_local,fig_on,request,Jain_request,Jain_path

    if Step3_choice == 0:
        choice = 'PF'
    if Step3_choice == 1:
        choice = 'PS'
    if Step3_choice == 2:
        choice = 'PU'
    
    print('[Performance Report] -- '+ choice)
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    print('Average Capacity Utilization:' +  str(round(average_capacity_utilization*100,2)) + '%')    
    print('Variance in Capacity Utilization:' +  str(round(variance_capacity_utilization,3)))    
    print('Weighted Throughput:' + str (round(weighted_flow_sum,2)))
    print('Weighted Min Throughput:' + str (round(weighted_flow_min,2)))
    print('Average Path Length:' + str (round(ave_path_length,2)))
    print('Average Flow Variance:' + str (round(ave_var_flow*100,2)) + '%')
    print('Fairness over paths:' + str (round(Jain_path,2)))
    print('Fairness over requests:' + str (round(Jain_request,2)))
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    
    # Plot Traffic
    if fig_on == 1:
        
        plt.figure(figsize=(6,6)) 

        traffic_color = []   
        nx.set_edge_attributes(G, traffic_color, 'traffic_color')

        traffic_width = []   
        nx.set_edge_attributes(G, traffic_width, 'traffic_width')

        node_color = []   
        nx.set_node_attributes(G, node_color, 'node_color')

        # grid pos
        for di in G.nodes():
            pos[di] = [di[0],di[1]]

        for k in G.edges():
            if G.edges[k]['traffic_on_edge'] == []:
                G.edges[k]['traffic_color'] = 'k'
            elif G.edges[k]['traffic_on_edge'] < 30:
                G.edges[k]['traffic_color'] = 'g'
            elif G.edges[k]['traffic_on_edge'] < 70:
                G.edges[k]['traffic_color'] = 'y'
            else:
                G.edges[k]['traffic_color'] = 'r'

            if G.edges[k]['traffic_on_edge'] == []:
                G.edges[k]['traffic_width'] = 3
            else:
                G.edges[k]['traffic_width'] = max(G.edges[k]['traffic_on_edge']/8,3)  

        nx.draw(G, pos, edge_color = [G.edges[k]['traffic_color'] for k in G.edges()] , node_color = 'w', with_labels=False, font_weight='light', node_size= 100, width= [G.edges[k]['traffic_width'] for k in G.edges()] )        

        for k in G.nodes():        
            if k in [request[m][0] for m in range(len(request))]:
                G.nodes[k]['node_color'] = 'r'
            elif k in [request[m][1] for m in range(len(request))]:
                G.nodes[k]['node_color'] = 'b'
            else:
                G.nodes[k]['node_color'] = 'w'
            nx.draw_networkx_nodes(G, pos, nodelist =[k], edgecolors ='k',node_color = G.nodes[k]['node_color'], node_size = 250)

        plt.show()

        plt.figure(figsize=(6,6)) 
        list_utilization = [G.edges[k]['traffic_on_edge'] for k in G.edges()]
        list_utilization = list(filter(None, list_utilization))
        list_utilization.sort()

        plt.plot(list_utilization,'bs')
        plt.plot([average_capacity_utilization*100]*len(list_utilization),'r--')
        plt.show()

def Step3_Capacity_Allocation_PS(Step3_output):

    global G,max_num_path,num_request

    for k in list(G.edges):
        edge_capacity = G.edges[k]['capacity']
        edge_request = G.edges[k]['request_ID_on_edge']

        # Restrict the number of paths on every edge
        if len(edge_request) > max_num_path:
            num_cancel = len(edge_request) - max_num_path

            cancel_count = 0
            singular_request = []
            
            while cancel_count < num_cancel:               # recursive cancellation
                break_flag = 0
                
                for c in range(len(edge_request)):
                    
                    if edge_request[c][0] in singular_request:
                        continue
                    
                    if edge_request[c][2] == max([h[2] for h in [hh for hh in edge_request if hh[0] not in singular_request]]):
                        
                        if [h[0] for h in edge_request].count(edge_request[c][0]) > 1:
                            edge_request.remove(edge_request[c])   # change in node attribute 'request_ID_on_edge' as well
                            cancel_count = cancel_count + 1
                            break_flag = 1
                        else:
                            singular_request.append(edge_request[c][0])
                            
                    if break_flag == 1:
                        break
            
        if len(edge_request) > max_num_path:              # Check: max path number realized
            print('Warning! Excessive Number of Path on edge ' + str(k))
            print(edge_request)
            print(remove_temp)
            break

        # ready for allocation

        edge_flow = list(edge_request).copy()    

        request_vector = list(np.zeros(num_request, dtype=int))

        ## P1 - Allocation between each request 

        # weighted #path vector for each request
        for ef in edge_flow:
            request_vector[ef[0]-1] += 1

        for n in range(num_request):
            if request_vector[n] != 0:
                request_vector[n] = request_vector[n]**beta          ## inverse

        weighted_vector = list(np.divide(request_vector,weight))   ## weight high, allocation low

        for n in range(num_request):
            if weighted_vector[n] != 0:
                weighted_vector[n] = weighted_vector[n]**-1          ## inverse

        # total channel for each request
        
        if sum(weighted_vector) != 0:
            channel_for_request = [np.floor(edge_capacity*weighted_vector[n]/sum(weighted_vector)) for n in range(num_request)]
        else:
            channel_for_request = [0]*num_request
            
        ## P2 - Allocation within each request
        temp_edge_request = []
        for i in range(len(edge_request)):
            temp_edge_request.append([edge_request[i][0], edge_request[i][2]**alpha])            ## power of alpha

        for i in range(len(edge_request)):
            ind_request = edge_request[i][0]  ## index of the request

        # the total alpha_weight of each request
        alpha_request_vector = list(np.zeros(num_request, dtype=int))

        for n in range(num_request):

            sum_alpha = 0
            for j in temp_edge_request:

                if j[0] == n+1:
                    sum_alpha = sum_alpha + j[1]

            alpha_request_vector[n] = sum_alpha

        flow_on_edge_raw = []
        for i in range(len(edge_request)):
            ind_request = edge_request[i][0]-1  ## index of the request

            flow_on_edge_raw.append\
            (max(f_min,np.floor(channel_for_request[ind_request]*temp_edge_request[i][1]/alpha_request_vector[ind_request])))  # fmin applied  

        ## P3 - Outbound Prevention

        while sum(flow_on_edge_raw) > edge_capacity:    # 这里改掉了之后利用率降低了约5%
            diff = sum(flow_on_edge_raw) - edge_capacity
            max_flow = max(flow_on_edge_raw)
            diff_divisor = flow_on_edge_raw.count(max_flow)

            for i in range(len(flow_on_edge_raw)):
                if flow_on_edge_raw[i] == max_flow:
                    flow_on_edge_raw[i] = flow_on_edge_raw[i] - np.ceil(diff/diff_divisor)
                    flow_on_edge_raw[i] = max(flow_on_edge_raw[i],f_min)

        G.edges[k]['flow_on_edge'] =  flow_on_edge_raw
        
        if Step3_output == 1:
            print('Edge'+ str(k) + '---' + str(flow_on_edge_raw))
            
def Step3_Capacity_Allocation_PU(Step3_output):
    
    global G,all_path_pool,max_num_path,num_request

    ## P1 -- Initialize max_flow vector for each path

    max_flow_on_path = []

    for ii in range(len(all_path_pool)):                  # for each request

        max_flow_on_path.append([])

        for jj in range(len(all_path_pool[ii][0])):       # for each path

            temp_vec = [min_capacity]*(len(all_path_pool[ii][0][jj])-1) 
            max_flow_on_path[ii].append(temp_vec)

    ## P2 -- Allocation

    count_no_improvement = 0       # iteration until no further change!   

    head_count = 0   # number of iterations
    
    while count_no_improvement < len(G.edges()): 
        
        k = list(G.edges())[head_count%len(G.edges())]
        
        head_count = head_count + 1
       
        edge_capacity = G.edges[k]['capacity']
        edge_request = G.edges[k]['request_ID_on_edge']
        
        if edge_request == []:    # Unused edge
            count_no_improvement = count_no_improvement + 1
            continue

        # Restrict the number of paths on every edge
        if len(edge_request) > max_num_path:
            num_cancel = len(edge_request) - max_num_path

            cancel_count = 0
            singular_request = []
            
            while cancel_count < num_cancel:               # recursive cancellation
                break_flag = 0
                
                for c in range(len(edge_request)):
                    
                    if edge_request[c][0] in singular_request:
                        continue
                    
                    if edge_request[c][2] == max([h[2] for h in [hh for hh in edge_request if hh[0] not in singular_request]]):
                        
                        if [h[0] for h in edge_request].count(edge_request[c][0]) > 1:
                            edge_request.remove(edge_request[c])   # change in node attribute 'request_ID_on_edge' as well
                            cancel_count = cancel_count + 1
                            break_flag = 1
                        else:
                            singular_request.append(edge_request[c][0])
                            
                    if break_flag == 1:
                        break
            
        if len(edge_request) > max_num_path:              # Check: max path number realized
            print('Warning! Excessive Number of Path on edge ' + str(k))
            break


        total_max_flow = 0                               # total max flow desired       --- edge specific 
        request_max_flow = [0]*num_request               # max flow on each request     --- edge specific 
        path_count = [0]*num_request                     # path count on each request   --- edge specific 
        max_flow_on_path_edge_specific = [[]]*num_request# max flow on each path        --- edge specific 

        for n in range(num_request):
            max_flow_on_path_edge_specific[n] = [0]*len(all_path_pool[n][0]) # All paths are counted. This is a sparse matrix.

        for ii in range(len(edge_request)):

            ind1 = edge_request[ii][0]-1                 # which request
            ind2 = edge_request[ii][1]-1                 # which path
            ind3 = edge_request[ii][3]                   # which order
            
            total_max_flow = total_max_flow + max_flow_on_path[ind1][ind2][ind3]
            request_max_flow[ind1] = request_max_flow[ind1] + max_flow_on_path[ind1][ind2][ind3]
            path_count[ind1] = path_count[ind1] + 1
            max_flow_on_path_edge_specific[ind1][ind2] = max_flow_on_path[ind1][ind2][ind3]
            
        # Total cancellation

        total_cancellation = max(total_max_flow - edge_capacity,0)

        # Stage 1 -- cancellation allocation on each request

        max_cancellation_request = [request_max_flow[tt] - f_min*path_count[tt] for tt in range(num_request)]
        weight_request = [tt**-1 for tt in weight]      # multiply weight^-1 = divide weight

        for nn in range(len(max_cancellation_request)): # non-used request, weight set to 0   
                if max_cancellation_request[nn] < 0:
                    weight_request[nn] = 0
                    max_cancellation_request[nn] = 0

        # prime function

        cancellation_on_request = Weighted_Adaptive_Cancellation(total_cancellation, list(max_cancellation_request), list(weight_request))    

        if sum(n < 0 for n in cancellation_on_request) > 0:
                print('Warning! Negative concellation on request ' + str(ind1+1))

        if sum(cancellation_on_request)!= total_cancellation:

            print('Warning! Sum of concellation allocation not match (Stage 1).')
            print('Allocated Cancellation:' + str(cancellation_on_request))    
            print('Total Cancellation:' + str(total_cancellation))    

        # Stage 2 -- cancellation allocation on each path

        cancellation_on_path = []
        for n in range(num_request):

            max_cancellation_path = [tt-f_min for tt in max_flow_on_path_edge_specific[n]]

            weight_path = [tt**(-alpha) for tt in all_path_pool[n][1]] 

            for nn in range(len(max_cancellation_path)): # un-used path, weight set to 0, max cancellation set to 0
                if max_cancellation_path[nn] < 0:
                    weight_path[nn] = 0
                    max_cancellation_path[nn] = 0

            # prime function      

            temp_allocation = Weighted_Adaptive_Cancellation(cancellation_on_request[n], list(max_cancellation_path), list(weight_path))
            cancellation_on_path.append(temp_allocation)    

            if sum(temp_allocation)!= cancellation_on_request[n]:  
                print('Warning! Sum of concellation allocation not match (Stage 2).')
                print('Allocated Cancellation:' + str(temp_allocation))    #-----------
                print('Total Cancellation:' + str(cancellation_on_request[n]))    #-----------

        # Stage 3 -- Make deduction

        flow_on_edge_raw = []                            
        for ii in range(len(edge_request)):
        
            ind1 = edge_request[ii][0]-1                 # which request
            ind2 = edge_request[ii][1]-1                 # which path
            ind3 = edge_request[ii][3]                   # which order
            
            if cancellation_on_path[ind1][ind2] < 0:
                print('Warning! Negative concellation on request ' + str(ind1+1) + ' path '+ str(ind2+1))

            temp_flow = max_flow_on_path[ind1][ind2][ind3] - cancellation_on_path[ind1][ind2]

            flow_on_edge_raw.append(int(temp_flow))

            # Update max_flow vector (only if further narrowing is needed)

            if temp_flow != max_flow_on_path[ind1][ind2][ind3]:    
                max_flow_on_path[ind1][ind2][ind3] = temp_flow
                
                if ind3 < len(max_flow_on_path[ind1][ind2])-1: # not last edge in the path
                    count_no_improvement = 0 # reset
                    
            for ii in range(len(max_flow_on_path[ind1][ind2])):   # max flow information updates on EVERY edge in the path
                    max_flow_on_path[ind1][ind2][ii] = min(temp_flow,max_flow_on_path[ind1][ind2][ii])
      
        # Allocation complete  

        if len(flow_on_edge_raw) != len(edge_request):
            print('Warning! Edge' + str(k) + ' has unallocated request.')

        if flow_on_edge_raw != G.edges[k]['flow_on_edge'] and ind3 < len(max_flow_on_path[ind1][ind2])-1: # not the last edge 
            
            G.edges[k]['flow_on_edge'] = flow_on_edge_raw
                
        else: # no improvement or last edge
            
            G.edges[k]['flow_on_edge'] = flow_on_edge_raw
            count_no_improvement = count_no_improvement + 1
            #G.edges[k]['allocation_complete'] = 1
            
        if sum(flow_on_edge_raw) > G.edges[k]['capacity']:
            print('Warning! Edge' + str(k) + ' exceeds capacity.')

        if Step3_output == 1:
            print('Edge'+ str(k) + '---' + str(flow_on_edge_raw))
    
    

# Main #
######################################################################################
import networkx as nx
from matplotlib import pyplot as plt
import math
import numpy as np
import random
import copy
import statistics as stat

from funcs import *
from params import *


print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('$$Routing Protocol for Quantum Transmission Network$$')
print('$$    Author:Tianyi Li    $$    tianyil@mit.edu    $$')
print('$$    Version:2020-09     $$                       $$')
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('$$ Please edit ''params.py'' to update parameters. $$')
    
######################################################################################
# Execution: Step 0 -- Initialization 

import settings

settings.init()

from settings import *

# Parameter window #
print('$$    Do you want to show input parameters: y/n    $$')
show_params = input()
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
if show_params == 'y':
    print('[Parameters]')
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    with open('params.py', 'r') as fin:
        for cnt, line in enumerate(fin):
            if line[0] not in ['#',' ',"\n","\t","\s","\v",'r','w']:
                print(line[:-1])
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    fin.close()


# Requests #
print('[Request]')
print(request)
print('[Average Request Distance]')
print(average_dis_request)
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
######################################################################################
# Execution: Step 1 -- Purification

if load_existing_lattice == 0:
    
    Step1_Capacity_Initialization(Step1_output)
    
    if save_new_lattice == 1:
        save_lattice_name = 'Lattice--' + save_run_name
        nx.write_gpickle(G,'output/' + save_lattice_name + '.gpickle')
else:
    
    G = nx.read_gpickle('output/' + load_lattice_name + '.gpickle')
    
    min_capacity = min([G.edges[k]['capacity'] for k in G.edges()])
    f_min = max(np.floor(min_capacity/max_num_path),1)
    pos = nx.spectral_layout(G)
    for di in G.nodes():
        pos[di] = [di[0],di[1]]
        
print('[Purification]')
print(str(len(G.edges)) + '/'+ str(2*Graph_width*(Graph_width-1)))
print('[Minimum Capacity]')
print(str(min_capacity))
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
######################################################################################
# Execution: Step 2-6 

Step2_Path_Determination()

## Progressive Filling ##  
Step3_choice = 0
Step3_Capacity_Allocation_PF(Step3_output)
Step4_Routing_Performance(Step4_output)
Step5_Capacity_Utilization(Step5_output)
Step6_Results()

## Proportional Share ##
Step3_choice = 1
Step3_Capacity_Allocation_PS(Step3_output)
Step4_Routing_Performance(Step4_output)
Step5_Capacity_Utilization(Step5_output)
Step6_Results()

## Propagatory Update ##
Step3_choice = 2
Step3_Capacity_Allocation_PU(Step3_output)
Step4_Routing_Performance(Step4_output)
Step5_Capacity_Utilization(Step5_output)
Step6_Results()

## Save Graph ##
if save_run == 1:
    nx.write_gpickle(G,'output/' + save_run_name + '.gpickle')
######################################################################################
