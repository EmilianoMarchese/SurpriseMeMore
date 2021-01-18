import numpy as np
from numba import jit
from . import auxiliary_function as AX


@jit(nopython=True)
def compute_sum(adj, raw_indices, column_indices):
    Sum = 0.0
    for ii in raw_indices:
        for jj in column_indices:
            Sum += adj[ii, jj]
    return Sum


def calculate_surprise_logsum_cp_weigh(adjacency_matrix,
                                       cluster_assignment,
                                       is_directed):
    """
    Computes core-periphery weighted surprise given a certain nodes'
    partitioning.
    """
    core_nodes = np.unique(np.where(cluster_assignment == 0)[0])
    periphery_nodes = np.unique(np.where(cluster_assignment == 1)[0])

    if is_directed:
        n_c = core_nodes.shape[0]
        n_x = periphery_nodes.shape[0]
        p_c = n_c * (n_c - 1)
        p_x = n_c * n_x * 2

        w_c = compute_sum(adjacency_matrix, core_nodes, core_nodes)
        w_x = compute_sum(adjacency_matrix,
                          core_nodes,
                          periphery_nodes) + compute_sum(adjacency_matrix,
                                                         periphery_nodes,
                                                         core_nodes)

        w = np.sum(adjacency_matrix)
        w_p = w - w_c - w_x
        n = n_c + n_x
        p = n * (n - 1)
        p_p = p - p_c - p_x

    else:
        n_c = core_nodes.shape[0]
        n_x = periphery_nodes.shape[0]
        p_c = n_c * (n_c-1) / 2
        p_x = n_c * n_x

        w_c = (compute_sum(adjacency_matrix, core_nodes, core_nodes))/2
        w_x = (compute_sum(adjacency_matrix,
                           core_nodes,
                           periphery_nodes) + compute_sum(adjacency_matrix,
                                                          periphery_nodes,
                                                          core_nodes))/2

        w = np.sum(adjacency_matrix)/2
        w_p = (w - w_c - w_x)/2
        n = n_c + n_x
        p = n * (n - 1) / 2
        p_p = p - p_c - p_x

    surprise = surprise_bipartite_logsum_CP_Weigh(p, p_c, p_x, p_p,
                                                  w, w_c, w_x, w_p)
    return surprise


@jit(nopython=True)
def surprise_bipartite_logsum_CP_Weigh(p, p_c, p_x, p_p, w, w_c, w_x, w_p):
    stop = False
    first_loop_break = False

    logP = logMultiHyperProbabilityWeight(p, p_c, p_x, p_p, w, w_c, w_x, w_p)
    for w_c_loop in range(w_c, w + 1):
        for w_x_loop in range(w_x, w + 1 - w_c_loop):
            if (w_c_loop == w_c) & (w_x_loop == w_x):
                continue
            w_p_loop = w - w_c_loop - w_x_loop
            nextLogP = logMultiHyperProbabilityWeight(p, p_c, p_x,
                                                      p_p, w, w_c_loop,
                                                      w_x_loop, w_p_loop)
            [logP, stop] = AX.sumLogProbabilities(nextLogP, logP)

            if stop:
                first_loop_break = True
                break
        if first_loop_break:
            break
    return -logP


@jit(nopython=True)
def logMultiHyperProbabilityWeight(p, p_c, p_x, p_p, w, w_c, w_x, w_p):
    logH = AX.logC(p_c+w_c-1, w_c) + AX.logC(p_x+w_x-1, w_x) + AX.logC(p_p+w_p-1, w_p) - AX.logC(p+w, w)
    return logH


def calculate_surprise_logsum_cp_bin(adjacency_matrix,
                                     cluster_assignment,
                                     is_directed):
    """
    Computes core-periphery binary surprise given a certain nodes'
    partitioning.
    """
    core_nodes = np.unique(np.where(cluster_assignment == 0)[0])
    periphery_nodes = np.unique(np.where(cluster_assignment == 1)[0])

    if is_directed:
        n_c = (core_nodes).shape[0]
        n_x = (periphery_nodes).shape[0]
        p_c = n_c * (n_c - 1)
        p_x = n_c * n_x * 2

        l_c = compute_sum(adjacency_matrix, core_nodes, core_nodes)
        l_x = compute_sum(adjacency_matrix, core_nodes, periphery_nodes) + compute_sum(adjacency_matrix,
                                                                                       periphery_nodes, core_nodes)

        l_t = np.sum(adjacency_matrix)
        n = n_c + n_x
        p = n * (n - 1)

    else: # is_directed == False : UNDIRECTED
        n_c = (core_nodes).shape[0]
        n_x = (periphery_nodes).shape[0]
        p_c = (n_c * (n_c - 1)) / (2)
        p_x = n_c * n_x

        l_c = compute_sum(adjacency_matrix, core_nodes, core_nodes)/2        
        l_x = (compute_sum(adjacency_matrix, core_nodes, periphery_nodes) + compute_sum(adjacency_matrix,
                                                                                       periphery_nodes, core_nodes))/2

        l_t = np.sum(adjacency_matrix)/2
        n = n_c + n_x
        p = (n*(n-1)) / 2

    if (p_c + p_x) < (l_c+l_x):
        return 0

    surprise = surprise_bipartite_logsum_CP_Bin(p, p_c, p_x, l_t, l_c, l_x)
    return surprise


@jit(nopython=True)
def surprise_bipartite_logsum_CP_Bin(p, p_c, p_x, l, l_c, l_x):
    stop = False
    first_loop_break = False

    min_l_p = min(l, p_c+p_x)
    
    logP = logMultiHyperProbability(p, p_c, p_x, l, l_c, l_x)
    for l_c_loop in range(l_c, min_l_p+1):
        for l_x_loop in range(l_x, min_l_p+1 - l_c_loop):
            if (l_c_loop == l_c) & (l_x_loop == l_x):
                continue
            nextLogP = logMultiHyperProbability(p, p_c, p_x, l, l_c_loop, l_x_loop)
            [logP, stop] = AX.sumLogProbabilities(nextLogP, logP)
    
            if stop:
                first_loop_break = True 
                break  
        if first_loop_break:
            break 
    
    return -logP


@jit(nopython=True)
def logMultiHyperProbability(p, p_c, p_x, l, l_c, l_x):
    logH = AX.logC(p_c, l_c) + AX.logC(p_x, l_x) + AX.logC(p - p_c - p_x, l-l_c-l_x) - AX.logC(p, l)
    return logH


def labeling_core_periphery(adjacency_matrix, cluster_assignment):
	#Function assigning the core and periphery labels based on link density

    core_nodes = np.where(cluster_assignment == 0)[0]
    periphery_nodes = np.where(cluster_assignment == 1)[0]
    l_core = np.sum(adjacency_matrix[np.ix_(list(core_nodes), list(core_nodes))] > 0)   
    l_periphery = np.sum(adjacency_matrix[np.ix_(list(periphery_nodes), list(periphery_nodes))] > 0)   
    core_density = l_core / (len(core_nodes)*(len(core_nodes)-1))
    periphery_density = l_periphery / (len(periphery_nodes)*(len(periphery_nodes)-1))

    if periphery_density > core_density:
        cluster_assignment_new = 1- cluster_assignment
        return cluster_assignment_new
    return cluster_assignment