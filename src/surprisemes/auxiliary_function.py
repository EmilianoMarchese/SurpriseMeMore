import numpy as np
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

        edgelist: list or np.ndarray -  List of weighted edges, eache edge must
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
