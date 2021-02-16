import numpy as np
from numba import jit
import scipy
from scipy.special import comb
import networkx as nx
from . import comdet_functions as CD


def compute_degree(a, is_directed):
    """returns matrix *a* degree sequence.

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
        elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
            return np.sum(a > 0, 0).A1, np.sum(a > 0, 1).A1
    else:    
        if type(a) == np.ndarray:
            return np.sum(a > 0, 1)
        # if the matrix is a scipy sparse matrix
        elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
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
        elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
            return np.sum(a, 0).A1, np.sum(a, 1).A1
    else:
        # if the matrix is a numpy array
        if type(a) == np.ndarray:
            return np.sum(a, 1)
        # if the matrix is a scipy sparse matrix
        elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
            return np.sum(a, 1).A1


def from_edgelist(edgelist, is_sparse, is_directed):
    """Returns np.ndarray or scipy.sparse matrix from edgelist.

    :param edgelist: List of edges, eache edge must be given as a 2-tuples (u,v).
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
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edgelist)
    if is_sparse:
        return nx.to_scipy_sparse_matrix(G)
    else:
        return nx.to_numpy_array(G)


def from_weighted_edgelist(edgelist, is_sparse, is_directed):
    """Returns np.ndarray or scipy.sparse matrix from edgelist.

    :param edgelist: List of weighted edges, eache edge must be given as a 3-tuples (u,v,w).
    :type edgelist: [type]
    :param is_sparse: If true the returned matrix is sparse.
    :type is_sparse: bool
    :param is_directed: If true the graph is directed.
    :type is_directed: bool
    :return: Weighted adjacency matrix.
    :rtype: numpy.ndarray or scipy.sparse
    """
    if is_directed:
        G = nx.DiGraph()
    else:    
        G = nx.Graph()
    G.add_weighted_edges_from(edgelist)
    if is_sparse:
        return nx.to_scipy_sparse_matrix(G)
    else:
        return nx.to_numpy_array(G)


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
        raise TypeError("Adjacency matrix must be square. If you are passing an edgelist use the positional argument 'edgelist='.")
    if np.sum(adjacency < 0):
        raise ValueError(
            "The adjacency matrix entries must be positive."
                        )
    if (not check_symmetric(adjacency, is_sparse)) and (not is_directed):
        raise TypeError("The adjacency matrix seems to be not symmetric, we suggest to use 'DirectedGraphClass'.")


@jit(nopython=True, fastmath=True)
def sumLogProbabilities(nextLogP, logP):
    if nextLogP == 0:
        stop = True
    else:
        stop = False
        if nextLogP > logP:
            common = nextLogP
            diffExponent = logP - common
        else:
            common = logP
            diffExponent = nextLogP - common

        logP = common + ((np.log10(1 + 10**diffExponent)) / np.log10(10))

        if (nextLogP - logP) > -4:
            stop = True

    return logP, stop


@jit(nopython=True, fastmath=True)
def logC(n, k):

    if k == n:
        return 0
    elif (n > 1000) & (k > 1000):  # Stirling's binomial coeff approximation
        return logStirFac(n) - logStirFac(k) - logStirFac(n-k)
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
        return -n + n*np.log10(n) + np.log10(n*(1 + 4.0*n*(1.0 + 2.0*n)))/6.0 + np.log10(np.pi)/2.0


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
    for i in np.arange(xmin, xmax+1):
        csum += np.log10(i)
    return csum


@jit(nopython=True, fastmath=True)
def sumFactorial(n):
    csum = 0
    if n > 1:
        for i in np.arange(2, n+1):
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
    shuffled_edges = edges.astype(int)
    return shuffled_edges


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
    sorted_edges = sorted_edges.astype(np.int)
    return sorted_edges


def evaluate_surprise_clust_bin(adjacency_matrix,
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
        p = CD.intracluster_links(adjacency_matrix,
                                  cluster_assignment)
        p = int(p)
        # All the possible intracluster links
        M = CD.calculate_possible_intracluster_links(cluster_assignment,
                                                     is_directed)
        # Observed links
        m = np.sum(adjacency_matrix.astype(bool))
        # Possible links
        n = adjacency_matrix.shape[0]
        F = n*(n-1)
    else:
        # intracluster links
        p = CD.intracluster_links(adjacency_matrix,
                                  cluster_assignment)
        p = int(p/2)
        # All the possible intracluster links
        M = int(CD.calculate_possible_intracluster_links(cluster_assignment,
                                                         is_directed))
        # Observed links
        m = np.sum(adjacency_matrix.astype(bool))/2
        # Possible links
        n = adjacency_matrix.shape[0]
        F = int((n*(n-1))/2)

    surprise = surprise_hypergeometric(F, p, M, m)
    return surprise


def surprise_hypergeometric(F, p, M, m):
    surprise = 0
    min_p = min(M, m)
    for p_loop in np.arange(p, min_p+1):
        surprise += (comb(M, p_loop, exact=True) * comb(F-M, m-p_loop, exact=True)) / comb(F, m, exact=True)
    return surprise


def evaluate_surprise_clust_weigh(adjacency_matrix,
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
        w = CD.intracluster_links(adjacency_matrix,
                                  cluster_assignment)
        # intracluster possible links
        Vi = CD.calculate_possible_intracluster_links(cluster_assignment,
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
        w = CD.intracluster_links(adjacency_matrix,
                                  cluster_assignment)/2
        # intracluster possible links
        Vi = CD.calculate_possible_intracluster_links(cluster_assignment,
                                                      is_directed)
        # Total Weight
        W = np.sum(adjacency_matrix)/2
        # Possible links
        n = adjacency_matrix.shape[0]
        V = int((n*(n-1))/2)
        # extracluster links
        Ve = V-Vi

    surprise = surprise_negative_hypergeometric(Vi, w, Ve, W, V)
    return surprise


def surprise_negative_hypergeometric(Vi, w, Ve, W, V):
    """Computes the negative hypergeometric distribution.
    """
    surprise = 0
    for w_loop in range(w, W):
        surprise += ((comb(Vi+w_loop-1, w_loop, exact=True) * comb(Ve+W-w_loop, W-w_loop, exact=True)) /
                     comb(V+W, W, exact=True))
    return surprise
