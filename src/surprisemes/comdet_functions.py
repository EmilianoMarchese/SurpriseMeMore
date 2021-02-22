import numpy as np
from numba import jit
from numba.typed import List
from . import auxiliary_function as AX


def calculate_possible_intracluster_links(partitions,
                                          is_directed):
    """Computes the number of possible links, given nodes memberships.

    :param partitions: Nodes memberships.
    :type partitions: numpy.array
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Total number of possible edges.
    :rtype: float
    """
    _, counts = np.unique(partitions, return_counts=True)
    nr_poss_intr_clust_links = np.sum(counts * (counts-1))
    if is_directed:
        return nr_poss_intr_clust_links
    else:
        return nr_poss_intr_clust_links/2


@jit(nopython=True)
def calculate_possible_intracluster_links_new(partitions,
                                              is_directed):
    """Computes the number of possible links, given nodes memberships.
    Faster implementation compiled in "nopython" mode.

    :param partitions: Nodes memberships.
    :type partitions: numpy.array
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Total number of possible edges.
    :rtype: float
    """
    counts = np.bincount(partitions)
    nr_poss_intr_clust_links = np.sum(counts * (counts-1))
    if is_directed:
        return nr_poss_intr_clust_links
    else:
        return nr_poss_intr_clust_links/2


@jit(nopython=True)
def intracluster_links(adj, partitions):
    """Computes intracluster links or weights.

    :param adj: Adjacency matrix.
    :type adj: numpy.array
    :param partitions: Nodes memberships.
    :type partitions: numpy.array
    :return: Number of intra-cluster links/weights.
    :rtype: float
    """
    nr_intr_clust_links = 0
    clust_labels = np.unique(partitions)
    for lab in clust_labels:
        indices = np.where(partitions == lab)[0]
        nr_intr_clust_links += intracluster_links_aux(adj, indices)
    return nr_intr_clust_links


@jit(nopython=True)
def intracluster_links_new(adj,
                           clust_labels,
                           partitions):
    """Computes intracluster links or weights. New implementation

    :param adj: Adjacency matrix.
    :type adj: numpy.array
    :param clust_labels: Labels of changed clusters.
    :type clust_labels: numpy.array
    :param partitions: Nodes memberships.
    :type partitions: numpy.array                                  
    :return: Number of intra-cluster links/weights.                
    :rtype: float                                                  
    """
    # print(clust_labels, clust_labels.shape, partitions)
    nr_intr_clust_links = np.zeros(clust_labels.shape[0])
    for ii, lab in enumerate(clust_labels):
        indices = np.where(partitions == lab)[0]                   
        nr_intr_clust_links[ii] = intracluster_links_aux(adj, indices)
    return nr_intr_clust_links     


@jit(nopython=True)
def intracluster_links_aux(adj, indices):
    """Computes intra-cluster links or weights given nodes indices.

    :param adj: [description]
    :type adj: [type]
    :param indices: [description]
    :type indices: [type]
    :return: [description]
    :rtype: [type]
    """
    n_links = 0
    for ii in indices:
        for jj in indices:
            n_links += adj[ii, jj]
    return n_links


def calculate_surprise_logsum_clust_bin(adjacency_matrix,
                                        cluster_assignment,
                                        is_directed):
    """Calculates the logarithm of the surprise given the current partitions for a binary network.

    :param adjacency_matrix: Binary adjacency matrix.
    :type adjacency_matrix: numpy.array
    :param cluster_assignment: Nodes memberships.
    :type cluster_assignment: numpy.array
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.
    :rtype: float
    """
    if is_directed:
        # intracluster links
        p = intracluster_links(adjacency_matrix,
                               cluster_assignment)
        p = int(p)
        # All the possible intracluster links
        M = calculate_possible_intracluster_links(cluster_assignment,
                                                  is_directed)
        # Observed links
        m = np.sum(adjacency_matrix.astype(bool))
        # Possible links
        n = adjacency_matrix.shape[0]
        F = n*(n-1)
    else:
        # intracluster links
        p = intracluster_links(adjacency_matrix,
                               cluster_assignment)
        p = int(p/2)
        # All the possible intracluster links
        M = int(calculate_possible_intracluster_links(cluster_assignment,
                                                      is_directed))
        # Observed links
        m = np.sum(adjacency_matrix.astype(bool))/2
        # Possible links
        n = adjacency_matrix.shape[0]
        F = int((n*(n-1))/2)

    surprise = surprise_logsum_Clust_Bin(F, p, M, m)
    return surprise


@jit(nopython=True)
def calculate_surprise_logsum_clust_bin_new(adjacency_matrix,
                                            cluster_assignment,
                                            mem_intr_link,
                                            clust_labels,
                                            args,
                                            is_directed):
    """Calculates the logarithm of the surprise given the current partitions for a binary network.
    New faster implementation reducing the number of redundant computations.

    :param adjacency_matrix: Binary adjacency matrix.
    :type adjacency_matrix: numpy.array
    :param cluster_assignment: Nodes memberships.                                                       
    :type cluster_assignment: numpy.array
    :param mem_intr_link: Intracluster links per cluster
    :type mem_intr_link: np.array
    :param clust_labels: Labels of changed clusters.
    :type clust_labels: np.array
    :param args: (Observed links, nodes number, possible links)
    :type args: (int, int, int)
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.                                                                              
    :rtype: float                                                                                       
    """                                                                                                 
    if is_directed:                                                                                     
        # intracluster links                                                                            
        int_links = intracluster_links_new(adj=adjacency_matrix,
                                           clust_labels=clust_labels,
                                           partitions=cluster_assignment)
        
        for node_label, nr_links in zip(clust_labels, int_links):
            mem_intr_link[node_label] = nr_links
        
        p = np.sum(mem_intr_link)
        p = int(p)                                                                                      
        # All the possible intracluster links                                                           
        M = calculate_possible_intracluster_links_new(cluster_assignment,                                   
                                                      is_directed)                                          
        # Observed links                                                                                
        m = args[0] 
        # Possible links                                                                                
        n = args[1]
        F = args[2]
    else:                                                                                               
        # intracluster links                                                                            
        int_links = intracluster_links_new(adj=adjacency_matrix,
                                           clust_labels=clust_labels,
                                           partitions=cluster_assignment)
        
        for node_label, nr_links in zip(clust_labels, int_links):
            mem_intr_link[node_label] = nr_links
        
        p = np.sum(mem_intr_link)
        p = int(p/2)                                                                                    
        # All the possible intracluster links                                                           
        M = int(calculate_possible_intracluster_links_new(cluster_assignment,                               
                                                          is_directed))                                     
        # Observed links                                                                                
        m = args[0]/2                                                     
        # Possible links                                                                                
        n = args[1]
        F = int(args[2]/2)
                                                                                                        
    surprise = surprise_logsum_Clust_Bin(F, p, M, m)
    return surprise, mem_intr_link


@jit(nopython=True)
def surprise_logsum_Clust_Bin(F, p, M, m):
    """[summary]

    :param F: [description]
    :type F: [type]
    :param p: [description]
    :type p: [type]
    :param M: [description]
    :type M: [type]
    :param m: [description]
    :type m: [type]
    :return: [description]
    :rtype: [type]
    """
    stop = False
    min_p = min(M, m)

    logP = logHyperProbability(F, p, M, m)
    for p_loop in np.arange(p, min_p+1):
        if (p_loop == p):
            continue
        nextLogP = logHyperProbability(F, p_loop, M, m)
        [logP, stop] = AX.sumLogProbabilities(nextLogP, logP)
        if stop:
            break

    return -logP


@jit(nopython=True)
def logHyperProbability(F, p, M, m):
    '''Evaluates logarithmic hypergeometric distribution'''
    logH = AX.logC(M, p) + AX.logC(F-M, m-p) - AX.logC(F, m)
    return logH


def calculate_surprise_logsum_clust_weigh(adjacency_matrix,
                                          cluster_assignment,
                                          is_directed):
    """Calculates the logarithm of the surprise given the current partitions for a weighted network.

    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.
    :type cluster_assignment: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.
    :rtype: float
    """
    if is_directed:
        # intracluster weights
        w = intracluster_links(adjacency_matrix,
                               cluster_assignment)
        # intracluster possible links
        Vi = calculate_possible_intracluster_links(cluster_assignment,
                                                   is_directed)
        # Total Weight
        W = np.sum(adjacency_matrix)
        # Possible links
        n = adjacency_matrix.shape[0]
        V = n*(n-1)
        # extracluster links
        Ve = V-Vi
    else:
        # intracluster weights
        w = intracluster_links(adjacency_matrix,
                               cluster_assignment)/2
        # intracluster possible links
        Vi = calculate_possible_intracluster_links(cluster_assignment,
                                                   is_directed)
        # Total Weight
        W = np.sum(adjacency_matrix)/2
        # Possible links
        n = adjacency_matrix.shape[0]
        V = int((n*(n-1))/2)
        # extracluster links
        Ve = V-Vi

    surprise = surprise_logsum_Clust_weigh(Vi, w, Ve, W, V)
    return surprise


@jit(nopython=True)
def surprise_logsum_Clust_weigh(Vi, w, Ve, W, V):
    """

    :param Vi: [description]
    :type Vi: [type]
    :param w: [description]
    :type w: [type]
    :param Ve: [description]
    :type Ve: [type]
    :param W: [description]
    :type W: [type]
    :param V: [description]
    :type V: [type]
    :return: [description]
    :rtype: [type]
    """
    stop = False

    logP = logNegativeHyperProbability(Vi, w, Ve, W, V)
    for w_loop in range(w, W):
        if (w_loop == w):
            continue
        nextLogP = logNegativeHyperProbability(Vi, w_loop, Ve, W, V)
        [logP, stop] = AX.sumLogProbabilities(nextLogP, logP)
        if stop:
            break

    return -logP


@jit(nopython=True)
def logNegativeHyperProbability(Vi, w, Ve, W, V):
    '''Evaluates logarithmic hypergeometric distribution'''
    logH = AX.logC(Vi+w-1, w) + AX.logC(Ve+W-w, W-w) - AX.logC(V+W, W)
    return logH


def labeling_communities(partitions):
    """Gives labels to communities from 0 to number of communities minus one.

    :param partitions: Nodes memberships.
    :type partitions: numpy.array
    :return: Re-labeled nodes memberships.
    :rtype: numpy.array
    """
    different_partitions = np.unique(partitions, return_counts=True)
    aux_argsort = np.argsort(different_partitions[1])[::-1]
    ordered_clusters = different_partitions[0][aux_argsort]
    new_partitioning = partitions.copy()
    new_partitions = np.array([k for k in range(len(ordered_clusters))])
    for old_part, new_part in zip(ordered_clusters, new_partitions):
        # print(old_part,new_part)
        indices_old = np.where(partitions == old_part)[0]
        new_partitioning[indices_old] = new_part

    return new_partitioning


def flipping_function_comdet(comm):
    """Changes the membership of a randomly selected node.

    :param comm: Nodes memberships.
    :type comm: numpy.array
    :return: New nodes memberships.
    :rtype: numpy.array
    """
    labels_set = np.unique(comm)
    node_index = np.random.randint(0, comm.shape[0])
    remaining_labels = labels_set[labels_set != comm[node_index]]
    if remaining_labels.size != 0:
        new_label = np.random.choice(remaining_labels)
        comm[node_index] = new_label
    return comm


def flipping_function_comdet_new(adj,
                                 membership,
                                 is_directed):
    
    obs_links = int(np.sum(adj.astype(bool)))
    n_nodes = int(adj.shape[0])                                    
    poss_links = int(n_nodes*(n_nodes-1))
    args = (obs_links, n_nodes, poss_links)
    
    surprise = calculate_surprise_logsum_clust_bin(adjacency_matrix=adj, cluster_assignment=membership, is_directed=is_directed)
    
    mem_intr_link = np.zeros(membership.shape[0], dtype=np.int32)
    # print(np.unique(membership), mem_intr_link, membership)
    for ii in np.unique(membership):
        indices = np.where(membership == ii)[0]
        mem_intr_link[ii] = intracluster_links_aux(adj, indices)

    # print("siamo qui")
    for node, node_label in zip(np.arange(membership.shape[0]), membership):

        for new_clust in np.unique(membership):
            if node_label != new_clust:
                aux_membership = membership.copy()
                aux_membership[node] = new_clust
                #print(np.array([node_label, new_clust]))
                temp_surprise, temp_mem_intr_link = calculate_surprise_logsum_clust_bin_new(adjacency_matrix=adj,
                                                                                            cluster_assignment=aux_membership,
                                                                                            mem_intr_link=mem_intr_link.copy(),
                                                                                            clust_labels=np.array([node_label, new_clust]),
                                                                                            args=args,
                                                                                            is_directed=is_directed)
                if temp_surprise > surprise:
                    membership = aux_membership.copy()
                    surprise = temp_surprise
                    mem_intr_link = temp_mem_intr_link
    
    return membership

