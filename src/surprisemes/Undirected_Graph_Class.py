import numpy as np
from numba import jit, prange
from scipy import comb
import scipy
from auxiliary_function import *


class UndirectedGraph:
    def __init__(
        self,
        adjacency=None,
        edgelist=None,
    ):
        self.n_nodes = None
        self.n_edges = None
        self.adjacency = None
        self.is_sparse = False
        self.edgelist = None
        self.dseq = None
        self.strength_sequence = None
        self.nodes_dict = None
        self.is_initialized = False
        self.is_weighted = False
        self._initialize_graph(
            adjacency=adjacency,
            edgelist=edgelist,
        )

    def _initialize_graph(
        self,
        adjacency=None,
        edgelist=None,
    ):

        if adjacency is not None:
            if not isinstance(
                adjacency, (list, np.ndarray)
            ) and not scipy.sparse.isspmatrix(adjacency):
                raise TypeError(
                    "The adjacency matrix must be passed as a \
                         list or numpy array or scipy sparse matrix."
                )
            elif adjacency.size > 0:
                if np.sum(adjacency < 0):
                    raise TypeError(
                        "The adjacency matrix entries must be positive."
                    )
                if isinstance(
                    adjacency, list
                ):  # Cast it to a numpy array: if it is given as a list it should not be too large
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
                else:
                    self.adjacency = adjacency
                    self.is_sparse = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError(
                    "The edgelist must be passed as a list or numpy array."
                )
            elif len(edgelist) > 0:
                if len(edgelist[0]) > 3:
                    raise ValueError(
                        "This is not an edgelist. An edgelist must be \
                         a list or array of couples of nodes with optional \
                         weights. Is this an adjacency matrix?"
                    )
                elif len(edgelist[0]) == 2:
                    self.adjacency = from_edgelist(edgelist)
                else:
                    self.adjacency = from_weighted_edgelist(edgelist)
                self.n_nodes = len(self.dseq)
                self.n_edges = np.sum(self.dseq)/2
                self.is_initialized = True
                if self.n_nodes > 2000:
                    self.is_sparse = True
        else:
            raise TypeError("UndirectedGraph is missing one \
                            positional argument adjacency or edgelist.")

        if np.sum(adjacency) == np.sum(adjacency > 0):
            self.dseq = compute_degree(adjacency).astype(np.int64)
        else:
            self.dseq = compute_degree(adjacency).astype(np.int64)
            self.strength_sequence = compute_strength(adjacency).astype(
                np.float64
            )
            self.is_weighted = True

    def set_adjacency_matrix(self, adjacency):
        if self.is_initialized:
            print(
                "Graph already contains edges or has \
                 a degree sequence. Use clean_edges() first."
            )
        else:
            self._initialize_graph(adjacency=adjacency)

    def set_edgelist(self, edgelist):
        if self.is_initialized:
            print(
                "Graph already contains edges or has \
                 a degree sequence. Use clean_edges() first."
            )
        else:
            self._initialize_graph(edgelist=edgelist)

    def clean_edges(self):
        self.adjacency = None
        self.edgelist = None
