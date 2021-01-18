import numpy as np
from numba import jit
import scipy
import networkx as nx


def compute_degree(a):
    """returns matrix A degrees

    :param a: numpy.ndarray, a matrix
    :return: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a > 0, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a > 0, 1).A1


def compute_strength(a):
    """returns matrix A strengths

    :param a: numpy.ndarray, a matrix
    :return: numpy.ndarray
    """
    # if the matrix is a numpy array
    if type(a) == np.ndarray:
        return np.sum(a, 1)
    # if the matrix is a scipy sparse matrix
    elif type(a) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
        return np.sum(a, 1).A1


def from_edgelist(edgelist, is_sparse):
    """ Returns np.ndarray or scipy.sparse matrix
        from edgelist.

        edgelist: list or np.ndarray -  List of edges, eache edge must
                 be given as a 2-tuples (u,v).
        is_sparse: boolean - If true the function returns a scipy.sparse
                  matrix.
    """
    G = nx.Graph()
    G.add_edges_from(edgelist)
    if is_sparse:
        return nx.to_scipy_sparse_matrix(G)
    else:
        return nx.to_numpy_array(G)


def from_weighted_edgelist(edgelist, is_sparse):
    """ Returns np.ndarray or scipy.sparse matrix
        from edgelist.

        edgelist: list or np.ndarray - List of weighted edges, eache edge must
                 be given as a 3-tuples (u,v).
        is_sparse: boolean - If true the function returns a scipy.sparse
                 matrix.
    """
    G = nx.Graph()
    G.add_weighted_edges_from(edgelist)
    if is_sparse:
        return nx.to_scipy_sparse_matrix(G)
    else:
        return nx.to_numpy_array(G)


def check_symmetric(a, is_sparse, rtol=1e-05, atol=1e-08):
    if is_sparse:
        return np.all(np.abs(a - a.T) < atol)
    else:    
        return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_adjacency(adjacency, is_sparse):
    """ Functions checking the validty of the
        adjacency matrix and raising error if it isn't.

        adjacency: np.ndarray - Adjacency matrix.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise TypeError("Adjacency matrix must be square. If you are passing an edgelist use the positional argument 'edgelist='.")
    if np.sum(adjacency < 0):
        raise TypeError(
            "The adjacency matrix entries must be positive."
                        )
    if not check_symmetric(adjacency, is_sparse):
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
    adj = adjacency_matrix.astype(bool).astype(np.int16)
    if not is_directed:
        adj = np.triu(adj)
    edges = np.stack(adj.nonzero(), axis=-1)
    np.random.shuffle(edges)
    shuffled_edges = edges.astype(int)
    return shuffled_edges


def jaccard_sorted_edges(adjacency_matrix):
    G = nx.from_numpy_matrix(adjacency_matrix)
    jacc = nx.jaccard_coefficient(G, ebunch=G.edges())
    jacc_array = []
    for u, v, p in jacc:
        jacc_array += [[u,v,p]]
    jacc_array = np.array(jacc_array)
    jacc_array = jacc_array[jacc_array[:,2].argsort()][::-1]
    sorted_edges = jacc_array[:,0:2]
    sorted_edges = sorted_edges.astype(np.int)
    return sorted_edges

