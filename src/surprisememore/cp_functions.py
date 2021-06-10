import numpy as np
from numba import jit

from . import auxiliary_function as ax
from . import comdet_functions as cd


@jit(nopython=True)
def compute_sum(adj, row_indices, column_indices):
    """Computes number of links/total weight given nodes indices.

    :param adj: Adjacency matrix.
    :type adj: numpy.ndarray
    :param row_indices: Row indices.
    :type row_indices: numpy.ndarray
    :param column_indices: Columns indices.
    :type column_indices: numpy.ndarray
    :return: Total links/weights
    :rtype: float
    """
    Sum = 0.0
    for ii in row_indices:
        for jj in column_indices:
            Sum += adj[ii, jj]
    return Sum


def calculate_surprise_logsum_cp_weigh(adjacency_matrix,
                                       cluster_assignment,
                                       is_directed):
    """Computes core-periphery weighted log-surprise given a certain nodes'
     partitioning.

    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Core periphery assigments.
    :type cluster_assignment: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise
    :rtype: float
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
        p_c = n_c * (n_c - 1) / 2
        p_x = n_c * n_x

        w_c = (compute_sum(adjacency_matrix, core_nodes, core_nodes)) / 2
        w_x = (compute_sum(adjacency_matrix,
                           core_nodes,
                           periphery_nodes) + compute_sum(adjacency_matrix,
                                                          periphery_nodes,
                                                          core_nodes)) / 2

        w = np.sum(adjacency_matrix) / 2
        w_p = (w - w_c - w_x) / 2
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
            [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)

            if stop:
                first_loop_break = True
                break
        if first_loop_break:
            break
    return -logP


@jit(nopython=True)
def logMultiHyperProbabilityWeight(p, p_c, p_x, p_p, w, w_c, w_x, w_p):
    """Computes the logarithm of the Negative Multinomial Hypergeometric
     distribution."""
    logH = (ax.logc(p_c + w_c - 1, w_c) + ax.logc(p_x + w_x - 1, w_x)
            + ax.logc(p_p + w_p - 1, w_p) - ax.logc(p + w, w))
    return logH


def calculate_surprise_logsum_cp_bin(adjacency_matrix,
                                     cluster_assignment,
                                     is_directed):
    """Computes core-periphery binary log-surprise given a certain nodes'
     partitioning.

    :param adjacency_matrix: Binary adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Core periphery assigments.
    :type cluster_assignment: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise
    :rtype: float
    """
    core_nodes = np.unique(np.where(cluster_assignment == 0)[0])
    periphery_nodes = np.unique(np.where(cluster_assignment == 1)[0])

    if is_directed:
        n_c = core_nodes.shape[0]
        n_x = periphery_nodes.shape[0]
        p_c = n_c * (n_c - 1)
        p_x = n_c * n_x * 2

        l_c = compute_sum(adjacency_matrix, core_nodes, core_nodes)
        l_x = compute_sum(adjacency_matrix, core_nodes,
                          periphery_nodes) + compute_sum(adjacency_matrix,
                                                         periphery_nodes,
                                                         core_nodes)

        l_t = np.sum(adjacency_matrix)
        n = n_c + n_x
        p = n * (n - 1)

    else:
        n_c = core_nodes.shape[0]
        n_x = periphery_nodes.shape[0]
        p_c = (n_c * (n_c - 1)) / (2)
        p_x = n_c * n_x

        l_c = compute_sum(adjacency_matrix, core_nodes, core_nodes) / 2
        l_x = (compute_sum(adjacency_matrix, core_nodes,
                           periphery_nodes) + compute_sum(adjacency_matrix,
                                                          periphery_nodes,
                                                          core_nodes)) / 2

        l_t = np.sum(adjacency_matrix) / 2
        n = n_c + n_x
        p = (n * (n - 1)) / 2

    if (p_c + p_x) < (l_c + l_x):
        return 0

    surprise = surprise_bipartite_logsum_CP_Bin(p, p_c, p_x, l_t, l_c, l_x)
    return surprise


@jit(nopython=True)
def surprise_bipartite_logsum_CP_Bin(p, p_c, p_x, l, l_c, l_x):
    stop = False
    first_loop_break = False

    # min_l_p = min(l, p_c + p_x)

    logP = logMultiHyperProbability(p, p_c, p_x, l, l_c, l_x)
    for l_c_loop in range(l_c, p_c + 1):
        for l_x_loop in range(l_x, p_x + 1):
            if ((l_c_loop == l_c) & (l_x_loop == l_x) or
                    (l_c_loop + l_x_loop > l)):
                continue
            nextLogP = logMultiHyperProbability(p, p_c, p_x,
                                                l, l_c_loop,
                                                l_x_loop)
            [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)

            if stop:
                first_loop_break = True
                break
        if first_loop_break:
            break

    return -logP


@jit(nopython=True)
def logMultiHyperProbability(p, p_c, p_x, l, l_c, l_x):
    """Computes the logarithm of the Multinomial Hypergeometric
     distribution."""
    logH = ax.logc(p_c, l_c) + ax.logc(p_x, l_x) + ax.logc(
        p - p_c - p_x,
        l - l_c - l_x) - ax.logc(p, l)
    return logH


@jit(nopython=True)
def compute_sum_enh(adj, row_indices, column_indices):
    """Computes number number of links and weights given nodes indices.

    :param adj: Adjacency matrix.
    :type adj: numpy.ndarray
    :param row_indices: Row indices.
    :type row_indices: numpy.ndarray
    :param column_indices: Columns indices.
    :type column_indices: numpy.ndarray
    :return: Total links/weights
    :rtype: float
    """
    n_link = 0.0
    weigth = 0.0
    for ii in row_indices:
        for jj in column_indices:
            if adj[ii, jj]:
                weigth += adj[ii, jj]
                n_link += 1
    return n_link, weigth


def calculate_surprise_logsum_cp_enhanced(adjacency_matrix,
                                          cluster_assignment,
                                          is_directed):
    """Computes core-periphery enhanced log-surprise given
     a certain nodes partitioning.

    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Core periphery assigments.
    :type cluster_assignment: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise
    :rtype: float
    """
    core_nodes = np.unique(np.where(cluster_assignment == 0)[0])
    periphery_nodes = np.unique(np.where(cluster_assignment == 1)[0])

    if is_directed:
        n_o = core_nodes.shape[0]
        n_p = periphery_nodes.shape[0]
        V_o = int(n_o * (n_o - 1))
        V_c = int(n_o * n_p * 2)

        l_o, w_o = compute_sum_enh(adjacency_matrix, core_nodes, core_nodes)
        l_c1, w_c1 = compute_sum_enh(adjacency_matrix,
                                     core_nodes,
                                     periphery_nodes)
        l_c2, w_c2 = compute_sum_enh(adjacency_matrix,
                                     periphery_nodes,
                                     core_nodes)

        l_o = int(l_o)
        w_o = int(w_o)
        l_c = int(l_c1 + l_c2)
        w_c = int(w_c1 + w_c2)
        L = int(np.sum(adjacency_matrix.astype(bool)))
        W = int(np.sum(adjacency_matrix))
        # w_p = W - w_o - w_c
        n = n_o + n_p
        V = int(n * (n - 1))

    else:
        n_o = core_nodes.shape[0]
        n_p = periphery_nodes.shape[0]
        V_o = int(n_o * (n_o - 1) / 2)
        V_c = int(n_o * n_p)

        l_o, w_o = (compute_sum_enh(adjacency_matrix, core_nodes, core_nodes))
        l_c1, w_c1 = compute_sum_enh(adjacency_matrix,
                                     core_nodes,
                                     periphery_nodes)
        l_c2, w_c2 = compute_sum_enh(adjacency_matrix,
                                     periphery_nodes,
                                     core_nodes)

        l_o = int(l_o / 2)
        w_o = int(w_o / 2)
        l_c = int((l_c1 + l_c2) / 2)
        w_c = int((w_c1 + w_c2) / 2)

        L = int(np.sum(adjacency_matrix.astype(bool)) / 2)
        W = int(np.sum(adjacency_matrix) / 2)
        # w_p = (W - w_o - w_c) / 2
        n = n_o + n_p
        V = int(n * (n - 1) / 2)

    # print("V_o", V_o, "l_o", l_o, "V_c", V_c, "l_c", l_c,
    #      "w_o", w_o, "w_c", w_c, "V", V, "L", L, "W", W)

    surprise = surprise_bipartite_logsum_CP_enh(V_o, l_o, V_c, l_c,
                                                w_o, w_c, V, L, W)
    return surprise


def surprise_bipartite_logsum_CP_enh(V_o, l_o, V_c, l_c, w_o, w_c, V, L, W):
    skip = False

    if (l_o > 0) and (l_o < L) and (l_c > 0) and (l_c < L):
        logP = log_enh_multy_hyper(V_o, l_o, V_c, l_c, w_o, w_c, V, L, W)
    elif (l_o == 0) and (l_c == L):
        logP = log_enh_red_univ_hyper(V_c, w_c,
                                      V, L, W)
        skip = True

    elif (l_o == L) and (l_c == 0):
        logP = log_enh_red_univ_hyper(V_o, w_o,
                                      V, L, W)
        skip = True
    elif (l_o == 0) and (l_c > 0) and (l_c < L):
        logP = cd.logenhancedhypergeometric(V_o=V_c, l_o=l_c,
                                            w_o=w_c, V=V,
                                            L=L, W=W)

    elif (l_o > 0) and (l_o < L) and (l_c == 0):
        logP = cd.logenhancedhypergeometric(V_o=V_o, l_o=l_o,
                                            w_o=w_o, V=V,
                                            L=L, W=W)
    else:
        logP = log_enh_red_univ_hyper(V - V_o - V_c,
                                      W - w_o - w_c,
                                      V, L, W)

    logP1 = logP
    logP2 = logP
    logP3 = logP
    l_c_loop = l_c
    w_o_loop = w_o
    w_c_loop = w_c

    for l_o_loop in range(l_o, V_o + 1):
        for l_c_loop in range(l_c, V_c + 1):
            if (l_o_loop + l_c_loop > L) or (l_o_loop + l_c_loop == 0):
                continue

            if ((l_o_loop > 0) and (l_o_loop < L) and
                    (l_c_loop > 0) and (l_c_loop < L)):
                w_o_loop, w_c_loop, logP, logP1 = case_one(
                    logP, logP1,
                    l_o_loop, l_c_loop,
                    V_o, l_o, V_c,
                    l_c, w_o, w_c,
                    V, L, W)
                nextLogP2 = log_enh_multy_hyper(V_o, l_o_loop,
                                                  V_c, l_c_loop,
                                                  w_o_loop, w_c_loop,
                                                  V, L, W)
                [logP2, stop2] = ax.sumLogProbabilities(nextLogP2, logP2)
                if stop2:
                    break
            elif ((l_o_loop == 0) and (l_c_loop > 0) and
                  (l_c_loop < L)):
                w_loop, logP = case_two(logP, l_c_loop, V_c,
                                        l_c, w_c, V, L, W)
                logP1 = logP
                nextLogP2 = cd.logenhancedhypergeometric(V_c, l_c_loop, w_loop,
                                                      V, L, W)

                [logP2, stop2] = ax.sumLogProbabilities(nextLogP2, logP2)
                if stop2:
                    break
            elif ((l_o_loop > 0) and (l_o_loop < L) and
                  (l_c_loop == 0)):

                continue

        if ((l_o_loop > 0) and (l_o_loop < L) and
                (l_c_loop > 0) and (l_c_loop < L)):

            nextLogP3 = log_enh_multy_hyper(V_o, l_o_loop,
                                            V_c, l_c_loop,
                                            w_o_loop, w_c_loop,
                                            V, L, W)
            [logP3, stop3] = ax.sumLogProbabilities(nextLogP3, logP3)
            if stop3:
                break
        elif ((l_o_loop == 0) and (l_c_loop > 0) and
              (l_c_loop < L)):
            continue
        elif ((l_o_loop > 0) and (l_o_loop < L) and
              (l_c_loop == 0)):
            w_loop, logP = case_two(logP, l_o_loop, V_o,
                                    l_o, w_o, V, L, W)
            logP1 = logP
            nextLogP3 = cd.logenhancedhypergeometric(V_o, l_o_loop,
                                                     w_loop, V, L, W)

            [logP3, stop3] = ax.sumLogProbabilities(nextLogP3, logP3)
            if stop3:
                break

    if skip:
        logP3 = logP

    return -logP3


@jit(nopython=True)
def case_one(logP, logP1, l_o_loop, l_c_loop, V_o,
             l_o, V_c, l_c, w_o, w_c, V, L, W):
    w_c_loop = w_c
    w_o_loop = w_o
    for w_o_loop in range(w_o - l_o_loop + l_o, W - L + l_o + 1):
        for w_c_loop in range(w_c - l_c_loop + l_c,
                              W + L + l_c + 1 - w_o_loop):
            if ((w_o_loop == w_o) & (w_c_loop == w_c) &
                    (l_o_loop == l_o) & (l_c_loop == l_c)):
                continue

            nextLogP = log_enh_multy_hyper(V_o, l_o_loop,
                                           V_c, l_c_loop,
                                           w_o_loop, w_c_loop,
                                           V, L, W)
            [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)
            if stop:
                break
        nextLogP1 = log_enh_multy_hyper(V_o, l_o_loop,
                                        V_c, l_c_loop,
                                        w_o_loop, w_c_loop,
                                        V, L, W)
        [logP1, stop1] = ax.sumLogProbabilities(nextLogP1, logP1)
        if stop1:
            break
    return w_o_loop, w_c_loop, logP, logP1


@jit(nopython=True)
def case_two(logP, l_loop, V_o,
             l_o, w_o, V, L, W):
    w_loop = w_o
    for w_loop in range(w_o - l_loop + l_o, W - L + l_o + 1):
        if (w_loop <= 0) & ((w_loop == w_o) & (l_loop == l_o)):
            continue
        nextLogP = cd.logenhancedhypergeometric(V_o, l_loop, w_loop, V, L, W)
        [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)
        if stop:
            break
    return w_loop, logP


@jit(nopython=True)
def log_enh_multy_hyper(V_o, l_o, V_c, l_c, w_o, w_c, V, L, W):
    """Computes the logarithm of the Negative Multinomial
     Hypergeometric distribution."""
    aux1 = (ax.logc(V_o, l_o) + ax.logc(V_c, l_c) + ax.logc(V - (V_o + V_c),
                                                            L - (l_o + l_c))) - ax.logc(V, L)
    aux2 = (ax.logc(w_o - 1, l_o - 1) + ax.logc(w_c - 1, l_c - 1) + ax.logc(
        W - (w_o + w_c) - 1, L - (l_o + l_c) - 1)) - ax.logc(W - 1, W - L)
    return aux1 + aux2


@jit(nopython=True)
def log_enh_red_univ_hyper(V_o, w_o, V, L, W):
    aux1 = ax.logc(V_o, L) - ax.logc(V, L)
    aux2 = ax.logc(w_o - 1, w_o - L)
    return aux1 + aux2


def labeling_core_periphery(adjacency_matrix, cluster_assignment):
    """Function assigning the core and periphery labels based on link density

    :param adjacency_matrix: [description]
    :type adjacency_matrix: [type]
    :param cluster_assignment: [description]
    :type cluster_assignment: [type]
    :return: [description]
    :rtype: [type]
    """
    core_nodes = np.where(cluster_assignment == 0)[0]
    periphery_nodes = np.where(cluster_assignment == 1)[0]
    l_core = np.sum(
        adjacency_matrix[np.ix_(list(core_nodes), list(core_nodes))] > 0)
    l_periphery = np.sum(adjacency_matrix[np.ix_(list(periphery_nodes),
                                                 list(periphery_nodes))] > 0)
    core_density = l_core / (len(core_nodes) * (len(core_nodes) - 1))
    periphery_density = l_periphery / (
            len(periphery_nodes) * (len(periphery_nodes) - 1))

    if periphery_density > core_density:
        cluster_assignment_new = 1 - cluster_assignment
        return cluster_assignment_new
    return cluster_assignment


def flipping_function_cp(comm, n_flipping):
    """Moves n nodes, randomly selected, from a partition to the other.

    :param comm: Nodes memberships.
    :type comm: numpy.ndarray
    :param n_flipping: Number of nodes to flipped.
    :type n_flipping: int
    :return: New nodes memberships.
    :rtype: numpy.ndarray
    """
    if np.random.random() > 0.5:
        comm[np.random.choice(np.where(comm == 1)[0],
                              replace=False,
                              size=n_flipping)] = 0
    else:
        comm[np.random.choice(np.where(comm == 0)[0],
                              replace=False,
                              size=n_flipping)] = 1
    return comm
