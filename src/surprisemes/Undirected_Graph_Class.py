import numpy as np
import scipy
from . import auxiliary_function as AX
from . import solver
from . import cp_functions as CP


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
                    "The adjacency matrix must be passed as a list or numpy array or scipy sparse matrix."
                )
            if isinstance(
                adjacency, list
            ):
                self.adjacency = np.array(adjacency)
            elif isinstance(
                adjacency, np.ndarray
            ):
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
                if len(edgelist[0]) == 2:
                    self.adjacency = AX.from_edgelist(edgelist, False)
                    self.edgelist = edgelist
                elif len(edgelist[0]) == 3:
                    self.adjacency = AX.from_weighted_edgelist(edgelist, False)
                    self.edgelist = edgelist
                else:
                    raise ValueError(
                        "This is not an edgelist. An edgelist must be a list or array of couples of nodes with optional weights. Is this an adjacency matrix?"
                    )
        else:
            raise TypeError("UndirectedGraph is missing one positional argument adjacency.")

        AX.check_adjacency(self.adjacency, False)
        if np.sum(self.adjacency) == np.sum(self.adjacency > 0):
            self.dseq = AX.compute_degree(self.adjacency).astype(np.int64)
        else:
            self.dseq = AX.compute_degree(self.adjacency).astype(np.int64)
            self.strength_sequence = AX.compute_strength(
                self.adjacency
                ).astype(np.float64)
            self.adjacency_weighted = self.adjacency
            self.adjacency = (self.adjacency_weighted > 0).astype(np.int16)
            self.is_weighted = True
        self.n_nodes = len(self.dseq)
        self.n_edges = int(np.sum(self.dseq)/2)
        self.is_initialized = True

    def set_adjacency_matrix(self, adjacency):
        if self.is_initialized:
            raise ValueError(
                "Graph already contains edges or has a degree sequence. Use 'clean_edges()' first."
            )
        else:
            self._initialize_graph(adjacency=adjacency)

    def set_edgelist(self, edgelist):
        if self.is_initialized:
            raise ValueError(
                "Graph already contains edges or has a degree sequence. Use 'clean_edges()' first."
            )
        else:
            self._initialize_graph(edgelist=edgelist)

    def clean_edges(self):
        self.adjacency = None
        self.edgelist = None
        self.is_initialized = False

    def run_cp_detection(self,
                         weighted = None,
                         num_sim=1,
                         sorting_method="default"):
    
        self.initialize_problem(weighted=weighted,
                                num_sim=num_sim,
                                sorting_method=sorting_method)

        sol = solver.solver_cp(adjacency_matrix = self.aux_adj,
                               num_sim = num_sim,
                               sort_edges = self.sorting_function,
                               calculate_surprise = self.surprise_function,
                               correct_partition_labeling = self.partition_labeler,
                               is_directed=True,
                               print_output=False)

        self.set_solved_problem(sol)

    def initialize_problem(self,
                           weighted,
                           num_sim,
                           sorting_method):
        if weighted is None:
            if self.is_weighted:
                self.aux_adj = self.adjacency_weighted
                self.method = "weighted"
            else:
                self.aux_adj = self.adjacency
                self.method = "binary"
        elif weighted:
            try:
                self.aux_adj =  self.adjacency_weighted
                self.method = "weighted"
            except:
                raise TypeError("You choose weighted core peryphery detection but the graph you initialised is binary.")
        else:
            self.aux_adj = self.adjacency
            self.method = "binary"
        
        sort_func = {
                     "random": lambda x: AX.shuffled_edges(x, False),
                     "jaccard": lambda x: AX.jaccard_sorted_edges(x),
                     "zmotifs": None,
                    }

        try:
            self.sorting_function[sorting_method]
        except:
            raise ValueError("Sorting method can be 'random', 'jaccard' and 'zmotifs'.")
        
        surp_fun = {
                    "binary": lambda x,y : CP.calculate_surprise_logsum_cp_bin(x, y, False),
                    "weighted": lambda x,y : CP.calculate_surprise_logsum_cp_weigh(x, y, False),
                    }
        
        try:
            self.surprise_function = surp_fun[self.method]
        except:
            raise ValueError("CP method can be 'binary' or 'weighted'.")

        self.partition_labeler = lambda x,y: CP.labeling_core_periphery(x,y)

    def set_solved_problem(self, sol):
        self.solution = sol[0]
        self.log_surprise = sol[1]
        self.surprise = np.exp(-self.log_surprise)
