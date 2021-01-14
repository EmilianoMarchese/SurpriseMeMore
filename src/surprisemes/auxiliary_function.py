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
