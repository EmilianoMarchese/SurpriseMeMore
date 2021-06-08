import networkx as nx
import numpy as np
import scipy
from numba import jit
from scipy.sparse import isspmatrix
from scipy.special import comb

from . import comdet_functions as cd
from . import cp_functions as cp


def compute_neighbours(adj):
    lista_neigh = []
    for ii in np.arange(adj.shape[0]):
        lista_neigh.append(adj[ii, :].nonzero()[0])
    return lista_neigh


@jit(nopython=True)
def compute_cn(adjacency):
    """ Computes common neighbours table, each entry i,j of this table is the
     number of common neighbours between i and j.

    :param adjacency: Adjacency matrix.
    :type adjacency: numpy.ndarray
    :return: Common neighbours table.
    :rtype: numpy.ndarray
    """
    cn_table = np.zeros_like(adjacency)
    for i in np.arange(adjacency.shape[0]):
        neighbour_i = (adjacency[i, :] + adjacency[:, i]).astype(np.bool_)
        for j in np.arange(i + 1, adjacency.shape[0]):
            neighbour_j = (adjacency[j, :] + adjacency[:, j]).astype(np.bool_)
            cn_table[i, j] = cn_table[j, i] = np.multiply(neighbour_i,
                                                          neighbour_j).sum()
    return cn_table


@jit(nopython=True)
def common_neigh_init_guess_strong(adjacency):
    """Generates a preprocessed initial guess based on the common neighbours
     of nodes. It makes a stronger aggregation of nodes based on
      the common neighbours similarity.

    :param adjacency: Adjacency matrix.
    :type adjacency: numpy.ndarray
    :return: Initial guess for nodes memberships.
    :rtype: np.array
    """
    cn_table = compute_cn(adjacency)
    memberships = np.array(
        [k for k in np.arange(adjacency.shape[0], dtype=np.int32)])
    argsorted = np.argsort(adjacency.astype(np.bool_).sum(axis=1))[::-1]
    for aux_node1 in argsorted:
        aux_tmp = memberships == aux_node1
        memberships[aux_tmp] = memberships[np.argmax(cn_table[aux_node1])]
    return memberships


@jit(nopython=True)
def common_neigh_init_guess_weak(adjacency):
    """Generates a preprocessed initial guess based on the common neighbours
     of nodes. It makes a weaker aggregation of nodes based on
      the common neighbours similarity.

    :param adjacency: Adjacency matrix.
    :type adjacency: numpy.ndarray
    :return: Initial guess for nodes memberships.
    :rtype: np.array
    """
    cn_table = compute_cn(adjacency)
    memberships = np.array(
        [k for k in np.arange(adjacency.shape[0], dtype=np.int32)])
    degree = (adjacency.astype(np.bool_).sum(axis=1)
              + adjacency.astype(np.bool_).sum(axis=0))
    avg_degree = np.mean(degree)
    argsorted = np.argsort(degree)[::-1]
    for aux_node1 in argsorted:
        if degree[aux_node1] >= avg_degree:
            aux_tmp = memberships == aux_node1
            memberships[aux_tmp] = memberships[np.argmax(cn_table[aux_node1])]
    return memberships


def eigenvector_init_guess(adjacency, is_directed):
    """Generates an initial guess for core periphery detection method: nodes
    with higher eigenvector centrality are in the core.

    :param adjacency: Adjacency matrix.
    :type adjacency: np.ndarray
    :param is_directed: True if the network is directed.
    :type is_directed: bool
    :return: Initial guess.
    :rtype: np.ndarray
    """
    # TODO: Vedere come funziona la parte pesata

    n_nodes = adjacency.shape[0]
    aux_nodes = int(np.ceil((n_nodes * 5) / 100))
    if is_directed:
        graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        centra = nx.eigenvector_centrality_numpy(graph)
        centra1 = np.array([centra[key] for key in centra])
        membership = np.ones_like(centra1, dtype=np.int32)
        membership[np.argsort(centra1)[::-1][:aux_nodes]] = 0

    else:
        graph = nx.from_numpy_array(adjacency, create_using=nx.Graph)
        centra = nx.eigenvector_centrality_numpy(graph)
        centra1 = np.array([centra[key] for key in centra])
        membership = np.ones_like(centra1, dtype=np.int32)
        membership[np.argsort(centra1)[::-1][:aux_nodes]] = 0

    return membership


def fixed_clusters_init_guess_cn(adjacency, n_clust):
    """ Generates an intial guess with a fixed number 'n' of clusters.
    Nodes are organised in clusters based on the number of common neighbors.
    The starting members of clusters are the 'n' nodes with higher
    degrees/strengths.

    :param adjacency: Adjacency matrix.
    :type adjacency: numpy.ndarray
    :param n_clust: Partitions number.
    :type n_clust: int
    :return: Initial guess.
    :rtype: numpy.ndarray
    """

    aux_memb = np.ones(adjacency.shape[0], dtype=np.int32) * (n_clust - 1)

    cn = compute_cn(adjacency)
    degree = adjacency.astype(np.bool_).sum(axis=1) + adjacency.astype(
        np.bool_).sum(axis=0)
    avg_degree = np.mean(degree)
    degree_indices_g = np.nonzero(degree > 2)[0]
    degree_indices_l = np.nonzero(degree <= 2)[0]

    arg_max = np.argmax(degree[degree_indices_g])
    clust_element = degree_indices_g[arg_max]

    cluster_count = 0
    while cluster_count != n_clust - 1:
        aux_memb[clust_element] = cluster_count
        degree_indices_g = np.delete(degree_indices_g, arg_max)
        if len(degree_indices_g) == 0:
            break
        arg_max = np.argmin(cn[clust_element][degree_indices_g])
        clust_element = degree_indices_g[arg_max]
        cluster_count += 1

    if np.unique(aux_memb).shape[0] < n_clust - 1:
        cluster_count += 1
        arg_max = np.argmax(degree[degree_indices_l])
        clust_element = degree_indices_l[arg_max]
        while cluster_count != n_clust - 1:
            aux_memb[clust_element] = cluster_count
            degree_indices_l = np.delete(degree_indices_l, arg_max)
            if len(degree_indices_l) == 0:
                raise ValueError(
                    "The number of clusters is higher thant the nodes number.")
            arg_max = np.argmin(cn[clust_element][degree_indices_l])
            clust_element = np.argmin(cn[clust_element][degree_indices_l])
            cluster_count += 1

    aux = np.nonzero(aux_memb == n_clust - 1)[0]
    np.random.shuffle(aux)
    for node in aux:
        if degree[node] < avg_degree:
            continue
        aux_list = np.nonzero(aux_memb != n_clust - 1)[0]
        node_index = aux_list[np.argmax(cn[node, aux_list])]
        if isinstance(node_index, np.ndarray):
            node_index = np.random.choice(node_index)
        aux_memb[node] = aux_memb[node_index]

    return aux_memb


def compute_degree(a, is_directed):
    """Returns matrix *a* degree sequence.

    :param a: Matrix.
    :type a:  numpy.ndarray
    :param is_directed: True if the matrix is directed.
    :type is_directed: bool
    :return: Degree sequence.
    :rtype: numpy.ndarray.
    """
    # if the matrix is a numpy array
    if is_directed:
        if type(a) == np.ndarray:
            return np.sum(a > 0, 0), np.sum(a > 0, 1)
        # if the matrix is a scipy sparse matrix
        elif isspmatrix(a):
            return np.sum(a > 0, 0).A1, np.sum(a > 0, 1).A1
    else:
        if type(a) == np.ndarray:
            return np.sum(a > 0, 1)
        # if the matrix is a scipy sparse matrix
        elif isspmatrix(a):
            return np.sum(a > 0, 1).A1


def compute_strength(a, is_directed):
    """Returns matrix *a* strength sequence.

    :param a: Matrix.
    :type a: numpy.ndarray
    :param is_directed: True if the matrix is directed.
    :type is_directed: bool
    :return: Strength sequence.
    :rtype: numpy.ndarray
    """
    if is_directed:
        # if the matrix is a numpy array
        if type(a) == np.ndarray:
            return np.sum(a, 0), np.sum(a, 1)
        # if the matrix is a scipy sparse matrix
        elif isspmatrix(a):
            return np.sum(a, 0).A1, np.sum(a, 1).A1
    else:
        # if the matrix is a numpy array
        if type(a) == np.ndarray:
            return np.sum(a, 1)
        # if the matrix is a scipy sparse matrix
        elif isspmatrix(a):
            return np.sum(a, 1).A1


def from_edgelist(edgelist, is_sparse, is_directed):
    """Returns np.ndarray or scipy.sparse matrix from edgelist.

    :param edgelist: List of edges, eache edge must be given as a 2-tuples
     (u,v).
    :type edgelist: list or numpy.ndarray
    :param is_sparse: If true the returned matrix is sparse.
    :type is_sparse: bool
    :param is_directed: If true the graph is directed.
    :type is_directed: bool
    :return: Adjacency matrix.
    :rtype: numpy.ndarray or scipy.sparse
    """
    # TODO: vedere che tipo di sparse e'
    if is_directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_edges_from(edgelist)
    if is_sparse:
        return nx.to_scipy_sparse_matrix(g)
    else:
        return nx.to_numpy_array(g)


def from_weighted_edgelist(edgelist, is_sparse, is_directed):
    """Returns np.ndarray or scipy.sparse matrix from edgelist.

    :param edgelist: List of weighted edges, eache edge must be given as a
     3-tuples (u,v,w).
    :type edgelist: [type]
    :param is_sparse: If true the returned matrix is sparse.
    :type is_sparse: bool
    :param is_directed: If true the graph is directed.
    :type is_directed: bool
    :return: Weighted adjacency matrix.
    :rtype: numpy.ndarray or scipy.sparse
    """
    if is_directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_weighted_edges_from(edgelist)
    if is_sparse:
        return nx.to_scipy_sparse_matrix(g)
    else:
        return nx.to_numpy_array(g)


def check_symmetric(a, is_sparse, rtol=1e-05, atol=1e-08):
    """Checks if the matrix is symmetric.

    :param a: Matrix.
    :type a: numpy.ndarray or scipy.sparse
    :param is_sparse: If true the matrix is sparse.
    :type is_sparse: bool
    :param rtol: Tuning parameter, defaults to 1e-05.
    :type rtol: float, optional
    :param atol: Tuning parameter, defaults to 1e-08.
    :type atol: float, optional
    :return: True if the matrix is symmetric.
    :rtype: bool
    """
    if is_sparse:
        return np.all(np.abs(a - a.T) < atol)
    else:
        return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_adjacency(adjacency, is_sparse, is_directed):
    """Functions checking the _validty_ of the adjacency matrix.

    :param adjacency: Adjacency matrix.
    :type adjacency: numpy.ndarray or scipy.sparse
    :param is_sparse: If true the matrix is sparse.
    :type is_sparse: bool
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :raises TypeError: Matrix not square.
    :raises ValueError: Negative entries.
    :raises TypeError: Matrix not symmetric.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise TypeError(
            "Adjacency matrix must be square. If you are passing an edgelist"
            " use the positional argument 'edgelist='.")
    if np.sum(adjacency < 0):
        raise ValueError(
            "The adjacency matrix entries must be positive."
        )
    if (not check_symmetric(adjacency, is_sparse)) and (not is_directed):
        raise TypeError(
            "The adjacency matrix seems to be not symmetric, we suggest to use"
            " 'DirectedGraphClass'.")


@jit(nopython=True, fastmath=True)
def sumLogProbabilities(nextlogp, logp):
    if nextlogp == 0:
        stop = True
    else:
        stop = False
        if nextlogp > logp:
            common = nextlogp
            diffexponent = logp - common
        else:
            common = logp
            diffexponent = nextlogp - common

        logp = common + ((np.log10(1 + 10 ** diffexponent)) / np.log10(10))

        if (nextlogp - logp) > -4:
            stop = True

    return logp, stop


@jit(nopython=True, fastmath=True)
def logc(n, k):
    if k == n:
        return 0
    elif (n > 1000) & (k > 1000):  # Stirling's binomial coeff approximation
        return logStirFac(n) - logStirFac(k) - logStirFac(n - k)
    else:
        t = n - k
        if t < k:
            t = k
        logC = sumRange(t + 1, n) - sumFactorial(n - t)
        return logC


@jit(nopython=True, fastmath=True)
def logStirFac(n):
    if n <= 1:
        return 1.0
    else:
        return -n + n * np.log10(n) + np.log10(
            n * (1 + 4.0 * n * (1.0 + 2.0 * n))) / 6.0 + np.log10(np.pi) / 2.0


@jit(nopython=True, fastmath=True)
def sumRange(xmin, xmax):
    """[summary]

    :param xmin: [description]
    :type xmin: [type]
    :param xmax: [description]
    :type xmax: [type]
    :return: [description]
    :rtype: [type]
    """
    csum = 0
    for i in np.arange(xmin, xmax + 1):
        csum += np.log10(i)
    return csum


@jit(nopython=True, fastmath=True)
def sumFactorial(n):
    csum = 0
    if n > 1:
        for i in np.arange(2, n + 1):
            csum += np.log10(i)
    return csum


def shuffled_edges(adjacency_matrix, is_directed):
    """Shuffles edges randomly.

    :param adjacency_matrix: Matrix.
    :type adjacency_matrix: numpy.ndarray
    :param is_directed: True if graph is directed.
    :type is_directed: bool
    :return: Shuffled edgelist.
    :rtype: mumpy.ndarray
    """
    adj = adjacency_matrix.astype(bool).astype(np.int16)
    if not is_directed:
        adj = np.triu(adj)
    edges = np.stack(adj.nonzero(), axis=-1)
    np.random.shuffle(edges)
    shuff_edges = edges.astype(np.int32)
    return shuff_edges


def jaccard_sorted_edges(adjacency_matrix):
    """Returns edges ordered based on jaccard index.

    :param adjacency_matrix: Matrix.
    :type adjacency_matrix: numpy.ndarray
    :return: Ordered edgelist.
    :rtype: numpy.ndarray
    """
    G = nx.from_numpy_matrix(adjacency_matrix)
    jacc = nx.jaccard_coefficient(G, ebunch=G.edges())
    jacc_array = []
    for u, v, p in jacc:
        jacc_array += [[u, v, p]]
    jacc_array = np.array(jacc_array)
    jacc_array = jacc_array[jacc_array[:, 2].argsort()][::-1]
    sorted_edges = jacc_array[:, 0:2]
    sorted_edges = sorted_edges.astype(np.int32)
    return sorted_edges


def evaluate_surprise_clust_bin(adjacency_matrix,
                                cluster_assignment,
                                is_directed):
    """Calculates the logarithm of the surprise given the current partitions
     for a binary network.

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
        p = cd.intracluster_links(adjacency_matrix,
                                  cluster_assignment)
        p = int(p)
        # All the possible intracluster links
        M = cd.calculate_possible_intracluster_links(cluster_assignment,
                                                     is_directed)
        # Observed links
        m = np.sum(adjacency_matrix.astype(bool))
        # Possible links
        n = adjacency_matrix.shape[0]
        F = n * (n - 1)
    else:
        # intracluster links
        p = cd.intracluster_links(adjacency_matrix,
                                  cluster_assignment)
        p = int(p / 2)
        # All the possible intracluster links
        M = int(cd.calculate_possible_intracluster_links(cluster_assignment,
                                                         is_directed))
        # Observed links
        m = np.sum(adjacency_matrix.astype(bool)) / 2
        # Possible links
        n = adjacency_matrix.shape[0]
        F = int((n * (n - 1)) / 2)

    surprise = surprise_hypergeometric(F, p, M, m)
    return surprise


def surprise_hypergeometric(F, p, M, m):
    surprise = 0
    min_p = min(M, m)
    for p_loop in np.arange(p, min_p + 1):
        surprise += (comb(M, p_loop, exact=True) * comb(
            F - M, m - p_loop,
            exact=True)) / comb(F,
                                m,
                                exact=True)
    return surprise


def evaluate_surprise_clust_weigh(adjacency_matrix,
                                  cluster_assignment,
                                  is_directed):
    """Calculates the logarithm of the surprise given the current partitions
     for a weighted network.

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
        w = cd.intracluster_links(adjacency_matrix,
                                  cluster_assignment)
        # intracluster possible links
        Vi = cd.calculate_possible_intracluster_links(cluster_assignment,
                                                      is_directed)
        # Total Weight
        W = np.sum(adjacency_matrix)
        # Possible links
        n = adjacency_matrix.shape[0]
        V = n * (n - 1)
        # extracluster links
        Ve = V - Vi
    else:
        # intracluster weights
        w = cd.intracluster_links(adjacency_matrix,
                                  cluster_assignment) / 2
        # intracluster possible links
        Vi = cd.calculate_possible_intracluster_links(cluster_assignment,
                                                      is_directed)
        # Total Weight
        W = np.sum(adjacency_matrix) / 2
        # Possible links
        n = adjacency_matrix.shape[0]
        V = int((n * (n - 1)) / 2)
        # extracluster links
        Ve = V - Vi

    surprise = surprise_negative_hypergeometric(Vi, w, Ve, W, V)
    return surprise


def surprise_negative_hypergeometric(Vi, w, Ve, W, V):
    """Computes the negative hypergeometric distribution.
    """
    surprise = 0
    for w_loop in range(w, W):
        surprise += ((comb(Vi + w_loop - 1, w_loop, exact=True) * comb(
            Ve + W - w_loop, W - w_loop, exact=True)) /
                     comb(V + W, W, exact=True))
    return surprise


def evaluate_surprise_cp_bin(adjacency_matrix,
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

        l_c = cp.compute_sum(adjacency_matrix, core_nodes, core_nodes)
        l_x = cp.compute_sum(adjacency_matrix, core_nodes,
                             periphery_nodes) + cp.compute_sum(
                                                         adjacency_matrix,
                                                         periphery_nodes,
                                                         core_nodes)

        l_t = np.sum(adjacency_matrix)
        n = n_c + n_x
        p = n * (n - 1)

    else:
        n_c = core_nodes.shape[0]
        n_x = periphery_nodes.shape[0]
        p_c = (n_c * (n_c - 1)) / 2
        p_x = n_c * n_x

        l_c = cp.compute_sum(adjacency_matrix, core_nodes, core_nodes) / 2
        l_x = (cp.compute_sum(adjacency_matrix, core_nodes,
                              periphery_nodes) + cp.compute_sum(
                                                          adjacency_matrix,
                                                          periphery_nodes,
                                                          core_nodes)) / 2

        l_t = np.sum(adjacency_matrix) / 2
        n = n_c + n_x
        p = (n * (n - 1)) / 2

    if (p_c + p_x) < (l_c + l_x):
        return 0

    surprise = surprise_bipartite_cp_bin(p, p_c, p_x, l_t, l_c, l_x)
    return surprise


@jit(forceobj=True)
def surprise_bipartite_cp_bin(p, p_c, p_x, l, l_c, l_x):
    surprise = 0
    aux_first = 0
    for l_c_loop in range(l_c, p_c + 1):
        aux_first_temp = aux_first
        for l_x_loop in range(l_x, p_x + 1):
            if l_c_loop + l_x_loop > l:
                continue
            aux = multihyperprobability(p, p_c, p_x,
                                        l, l_c_loop,
                                        l_x_loop)
            surprise += aux
            if surprise == 0:
                break
            if aux/surprise < 1e-3:
                break

        aux_first = surprise
        if aux_first - aux_first_temp:
            if ((aux_first - aux_first_temp) / aux_first) < 1e-3:
                # pass
                break

    return surprise


# @jit(nopython=True)
def multihyperprobability(p, p_c, p_x, l, l_c, l_x):
    """Computes the logarithm of the Multinomial Hypergeometric
     distribution."""
    logh = comb(p_c, l_c, True) * comb(p_x, l_x) + comb(
        p - p_c - p_x,
        l - l_c - l_x) - comb(p, l)
    return logh


def evaluate_surprise_cp_enh(adjacency_matrix,
                             cluster_assignment,
                             is_directed):
    """Computes core-periphery weighted surprise given a
     certain nodes' partitioning.

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
        V_o = n_o * (n_o - 1)
        V_c = n_o * n_p * 2

        l_o, w_o = cp.compute_sum_enh(adjacency_matrix, core_nodes, core_nodes)
        l_c, w_c = cp.compute_sum_enh(adjacency_matrix,
                                      core_nodes,
                                      periphery_nodes) + cp.compute_sum_enh(
            adjacency_matrix,
            periphery_nodes,
            core_nodes)
        L = np.sum(adjacency_matrix.astype(bool))
        W = np.sum(adjacency_matrix)
        # w_p = W - w_o - w_c
        n = n_o + n_p
        V = n * (n - 1)

    else:
        n_o = core_nodes.shape[0]
        n_p = periphery_nodes.shape[0]
        V_o = n_o * (n_o - 1) / 2
        V_c = n_o * n_p

        l_o, w_o = (cp.compute_sum_enh(adjacency_matrix, core_nodes, core_nodes))
        l_c1, w_c1 = cp.compute_sum_enh(adjacency_matrix,
                                        core_nodes,
                                        periphery_nodes)
        l_c2, w_c2 = cp.compute_sum_enh(adjacency_matrix,
                                        periphery_nodes,
                                        core_nodes)

        l_o = l_o / 2
        w_o = w_o / 2
        l_c = (l_c1 + l_c2) / 2
        w_c = (w_c1 + w_c2) / 2

        L = np.sum(adjacency_matrix.astype(bool)) / 2
        W = np.sum(adjacency_matrix) / 2
        # w_p = (W - w_o - w_c) / 2
        n = n_o + n_p
        V = n * (n - 1) / 2

    # print("V_o", V_o, "l_o", l_o, "V_c", V_c, "l_c", l_c,
    #      "w_o", w_o, "w_c", w_c, "V", V, "L", L, "W", W)

    surprise = surprise_bipartite_cp_enh(V_o, l_o, V_c, l_c,
                                         w_o, w_c, V, L, W)
    return surprise


@jit(forceobj=True)
def surprise_bipartite_cp_enh(V_o, l_o, V_c, l_c, w_o, w_c, V, L, W):
    surprise = 0

    min_l_o = min(L, V_o + V_c)
    aux_first = 0
    aux_second = 0
    aux_third = 0
    for l_o_loop in np.arange(l_o, min_l_o + 1):
        aux_first_temp = aux_first
        for l_c_loop in np.arange(l_c, min_l_o + 1 - l_o_loop):
            aux_second_temp = aux_second
            for w_o_loop in np.arange(w_o, W + 1):
                aux_third_temp = aux_third
                for w_c_loop in np.arange(w_c, W + 1 - w_o_loop):
                    aux = logmulti_hyperprobability_weightenh(
                                                            V_o, l_o_loop,
                                                            V_c, l_c_loop,
                                                            w_o_loop, w_c_loop,
                                                            V, L, W)
                    surprise += aux
                    # print(surprise)
                    if surprise == 0:
                        break
                    if aux / surprise < 1e-3:
                        break

                aux_third = surprise
                if aux_third - aux_third_temp:
                    if ((aux_third - aux_third_temp) / aux_third) < 1e-3:
                        # pass
                        break
                else:
                    break

            aux_second = aux_third
            if aux_second - aux_second_temp:
                if ((aux_second - aux_second_temp) / aux_second) < 1e-3:
                    # pass
                    break
            else:
                break

        aux_first = aux_second
        if aux_first - aux_first_temp:
            if ((aux_first - aux_first_temp) / aux_first) < 1e-4:
                # pass
                break
        else:
            break

    return surprise


@jit(forceobj=True)
def logmulti_hyperprobability_weightenh(V_o, l_o, V_c, l_c, w_o, w_c, V, L, W):
    """Computes the of the Negative Multinomial Hypergeometric
     distribution."""
    aux1 = (comb(V_o, l_o, exact=True) * comb(V_c, l_c, exact=True) * comb(
        V - (V_o + V_c), L - (l_o + l_c), exact=True)) / comb(V, L, exact=True)
    aux2 = (comb(w_o - 1, l_o - 1, exact=True) * comb(w_c - 1, l_c - 1,
                                                      exact=True) * comb(
        W - (w_o + w_c) - 1, L - (l_o + l_c) - 1, exact=True)) / comb(W - 1,
                                                                      W - L,
                                                                      exact=True)
    return aux1 * aux2


def evaluate_surprise_community_enh(
        adjacency_matrix,
        cluster_assignment,
        is_directed):
    """Calculates surprise given the current partitions for a weighted network.

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
        l_o, w_o = cd.intracluster_links_enh(adjacency_matrix,
                                             cluster_assignment)
        # intracluster possible links
        V_o = cd.calculate_possible_intracluster_links(
            cluster_assignment,
            is_directed)
        # Total Weight
        W = np.sum(adjacency_matrix)
        L = np.sum(adjacency_matrix.astype(bool))
        # Possible links
        n = adjacency_matrix.shape[0]
        V = n * (n - 1)
        # extracluster links
        # inter_links = V - V_o
    else:
        # intracluster weights
        l_o, w_o = cd.intracluster_links_enh(adjacency_matrix,
                                             cluster_assignment)
        l_o = l_o / 2
        w_o = w_o / 2
        # intracluster possible links
        V_o = cd.calculate_possible_intracluster_links(
            cluster_assignment,
            is_directed)
        # Total Weight
        W = np.sum(adjacency_matrix) / 2
        L = np.sum(adjacency_matrix.astype(bool)) / 2
        # Possible links
        n = adjacency_matrix.shape[0]
        V = int((n * (n - 1)) / 2)
        # extracluster links
        # inter_links = V - V_o

    # print("V_0", V_o, "l_0", l_o, "w_0", w_o, "V", V, "L", L, "W", W)

    surprise = surprise_clust_enh(V_o, l_o, w_o, V, L, W)
    return surprise


@jit(forceobj=True)
def surprise_clust_enh(V_o, l_o, w_o, V, L, W):
    min_l_loop = min(L, V_o)

    surprise = 0.0
    aux_first = 0.0
    # print("l_o", l_o, "min l", min_l_loop,"w_0", w_o, "W-L", W)
    for l_loop in range(l_o, min_l_loop + 1):
        aux_first_temp = aux_first
        for w_loop in range(w_o - l_loop + l_o, W - L + l_o + 1):
            if w_loop <= 0:
                continue
            # print(l_loop,  w_loop)
            aux = logenhancedhypergeometric(V_o, l_loop, w_loop, V, L, W)

            if np.isnan(aux):
                break
            surprise += aux
            # print(aux, surprise)
            if surprise == 0:
                break
            if aux / surprise <= 1e-3:
                break
        aux_first = surprise
        if aux_first - aux_first_temp:
            if ((aux_first - aux_first_temp) / aux_first) < 1e-4:
                # pass
                break
        else:
            break

    return surprise


@jit(forceobj=True)
def logenhancedhypergeometric(V_o, l_o, w_o, V, L, W):
    if l_o < L:
        aux1 = (comb(V_o, l_o, True) * comb(V - V_o, L - l_o, True)) / comb(V, L, True)
        aux2 = (comb(w_o - 1, w_o - l_o, True) * comb(W - w_o - 1, (W - L) - (w_o - l_o), True)) / comb(W - 1, W - L, True)
    else:
        aux1 = (comb(V_o, l_o, True) / comb(V, L, True))
        aux2 = comb(w_o - 1, w_o - L, True)
    return aux1 * aux2


def evaluate_surprise_com_det_continuous(
        adjacency_matrix,
        cluster_assignment,
        is_directed):
    """Calculates the logarithm of the continuous surprise given
     the current partitions for a weighted network.

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
        w = cd.intracluster_links(adjacency_matrix,
                                  cluster_assignment)
        # intracluster possible links
        poss_intr_links = cd.calculate_possible_intracluster_links(
            cluster_assignment,
            is_directed)
        # Total Weight
        tot_weights = np.sum(adjacency_matrix)
        # Possible links
        n = adjacency_matrix.shape[0]
        poss_links = n * (n - 1)
        # extracluster links
        # inter_links = poss_links - poss_intr_links
    else:
        # intracluster weights
        w = cd.intracluster_links(adjacency_matrix,
                                  cluster_assignment) / 2
        # intracluster possible links
        poss_intr_links = cd.calculate_possible_intracluster_links(
            cluster_assignment,
            is_directed)
        # Total Weight
        tot_weights = np.sum(adjacency_matrix) / 2
        # Possible links
        n = adjacency_matrix.shape[0]
        poss_links = int((n * (n - 1)) / 2)
        # extracluster links
        # inter_links = poss_links - poss_intr_links

    surprise = cd.continuous_surprise_clust(
        V=poss_links,
        W=tot_weights,
        w_o=w,
        V_o=poss_intr_links)

    return surprise
