import numpy as np
from numba import jit
from . import auxiliary_function as AX


def calculate_possible_intracluster_links(partitions,
                                          is_directed):
    """Computes the number of possible links, given nodes memberships.

    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
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
def intracluster_links(adj, partitions):
    """Computes intracluster links or weights.

    :param adj: Adjacency matrix.
    :type adj: numpy.ndarray
    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
    :return: Number of intracluster links/weights.
    :rtype: float
    """
    nr_intr_clust_links = 0
    clust_labels = np.unique(partitions)
    for lab in clust_labels:
        indices = np.where(partitions == lab)[0]
        nr_intr_clust_links += intracluster_links_aux(adj, indices)
    return nr_intr_clust_links


@jit(nopython=True)
def intracluster_links_aux(adj, indices):
    """Computes intracluster links or weights given nodes indices.

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
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.
    :type cluster_assignment: numpy.ndarray
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
    :type partitions: numpy.ndarray
    :return: Re-labeled nodes memberships.
    :rtype: numpy.ndarray
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
    :type comm: numpy.ndarray
    :return: New nodes memberships.
    :rtype: numpy.ndarray
    """
    labels_set = np.unique(comm)
    node_index = np.random.randint(0, comm.shape[0])
    remaining_labels = labels_set[labels_set != comm[node_index]]
    if remaining_labels.size != 0:
        new_label = np.random.choice(remaining_labels)
        comm[node_index] = new_label
    return comm
