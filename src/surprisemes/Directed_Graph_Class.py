import numpy as np
import scipy
from . import auxiliary_function as AX
from . import solver
from . import cp_functions as CP
from . import comdet_functions as CD


class DirectedGraph:
    def __init__(
        self,
        adjacency,
        edgelist=None,
    ):
        self.n_nodes = None
        self.n_edges = None
        self.adjacency = None
        self.is_sparse = False
        self.edgelist = None
        self.degree_sequence_out = None
        self.degree_sequence_in = None
        self.strength_sequence_out = None
        self.strength_sequence_in = None
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
                    self.adjacency = AX.from_edgelist(edgelist,
                                                      self.is_sparse,
                                                      True)
                    self.edgelist = edgelist
                elif len(edgelist[0]) == 3:
                    self.adjacency = AX.from_weighted_edgelist(edgelist,
                                                               self.is_sparse,
                                                               True)
                    self.edgelist = edgelist
                else:
                    raise ValueError(
                        "This is not an edgelist. An edgelist must be a list or array of couples of nodes with optional weights. Is this an adjacency matrix?"
                    )
        else:
            raise TypeError("UndirectedGraph is missing one positional argument adjacency.")

        AX.check_adjacency(self.adjacency, self.is_sparse, True)
        if np.sum(self.adjacency) == np.sum(self.adjacency > 0):
            self.degree_sequence_in, self.degree_sequence_out = AX.compute_degree(
                                                     self.adjacency,
                                                     True
                                                     )
            self.degree_sequence_in = self.degree_sequence_in.astype(np.int64)
            self.degree_sequence_out = self.degree_sequence_out.astype(np.int64)
        else:
            self.degree_sequence_in, self.degree_sequence_out = AX.compute_degree(
                                                     self.adjacency,
                                                     True
                                                     )
            self.degree_sequence_in = self.degree_sequence_in.astype(np.int64)
            self.degree_sequence_out = self.degree_sequence_out.astype(np.int64)

            self.strength_sequence_in, self.strength_sequence_out = AX.compute_strength(
                self.adjacency,
                True
                )
            self.strength_sequence_in = self.strength_sequence_in.astype(np.float64)
            self.strength_sequence_out = self.strength_sequence_out.astype(np.float64)

            self.adjacency_weighted = self.adjacency
            self.adjacency = (self.adjacency_weighted.astype(bool)).astype(np.int16)
            self.is_weighted = True
        self.n_nodes = len(self.degree_sequence_out)
        self.n_edges = int(np.sum(self.degree_sequence_out)/2)
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
                         initial_guess=None,
                         weighted=None,
                         num_sim=2,
                         sorting_method="default",
                         print_output=False):

        self._initialize_problem_cp(initial_guess=initial_guess,
                                    weighted=weighted,
                                    sorting_method=sorting_method)

        sol = solver.solver_cp(adjacency_matrix=self.aux_adj,
                               cluster_assignment=self.init_guess,
                               num_sim=num_sim,
                               sort_edges=self.sorting_function,
                               calculate_surprise=self.surprise_function,
                               correct_partition_labeling=self.partition_labeler,
                               print_output=print_output)

        self._set_solved_problem(sol)

    def _initialize_problem_cp(self,
                               initial_guess,
                               weighted,
                               sorting_method):

        self._set_initial_guess_cp(initial_guess)
        if weighted is None:
            if self.is_weighted:
                self.aux_adj = self.adjacency_weighted
                self.method = "weighted"
            else:
                self.aux_adj = self.adjacency
                self.method = "binary"
        elif weighted:
            try:
                self.aux_adj = self.adjacency_weighted
                self.method = "weighted"
            except:
                raise TypeError("You choose weighted core peryphery detection but the graph you initialised is binary.")
        else:
            self.aux_adj = self.adjacency
            self.method = "binary"

        if (sorting_method == "default") and (self.is_weighted):
            sorting_method = "random"
        elif (sorting_method == "default") and (not self.is_weighted):
            sorting_method = "jaccard"

        sort_func = {
                     "random": lambda x: AX.shuffled_edges(x, True),
                     "degrees": None,
                     "strengths": None,
                    }

        try:
            self.sorting_function = sort_func[sorting_method]
        except:
            raise ValueError("Sorting method can be 'random', 'degrees' or 'strengths'.")

        surp_fun = {
                    "binary": lambda x, y: CP.calculate_surprise_logsum_cp_bin(x, y, True),
                    "weighted": lambda x, y: CP.calculate_surprise_logsum_cp_weigh(x, y, True),
                    }

        try:
            self.surprise_function = surp_fun[self.method]
        except:
            raise ValueError("CP method can be 'binary' or 'weighted'.")

        self.partition_labeler = lambda x, y: CP.labeling_core_periphery(x, y)

    def _set_initial_guess_cp(self, initial_guess):
        if initial_guess is None:
            self.init_guess = np.ones(self.n_nodes, dtype=int)
            if self.is_weighted:
                self.init_guess[self.strength_sequence_out.argsort()[-3:]] = 0
            else:
                self.init_guess[self.degree_sequence_out.argsort()[-3:]] = 0
        elif isinstance(initial_guess, np.ndarray):
            self.init_guess = initial_guess
        elif isinstance(initial_guess, list):
            self.init_guess = np.array(initial_guess)

        if self.init_guess.shape[0] != self.n_nodes:
            raise ValueError("The length of the initial guess provided is different from the network number of nodes.")

    def run_comunity_detection(self,
                               initial_guess=None,
                               weighted=None,
                               num_sim=2,
                               prob_mix=0.1,
                               sorting_method="default",
                               print_output=False):

        self._initialize_problem_cd(initial_guess=initial_guess,
                                    weighted=weighted,
                                    sorting_method=sorting_method)

        sol = solver.solver_com_det(adjacency_matrix=self.aux_adj,
                                    cluster_assignment=self.init_guess,
                                    num_sim=num_sim,
                                    sort_edges=self.sorting_function,
                                    calculate_surprise=self.surprise_function,
                                    correct_partition_labeling=self.partition_labeler,
                                    prob_mix=prob_mix,
                                    print_output=print_output)

        self._set_solved_problem(sol)

    def _initialize_problem_cd(self,
                               initial_guess,
                               weighted,
                               sorting_method):

        self._set_initial_guess_cd(initial_guess)
        if weighted is None:
            if self.is_weighted:
                self.aux_adj = self.adjacency_weighted
                self.method = "weighted"
            else:
                self.aux_adj = self.adjacency
                self.method = "binary"
        elif weighted:
            try:
                self.aux_adj = self.adjacency_weighted
                self.method = "weighted"
            except:
                raise TypeError("You choose weighted comunity detection but the graph you initialised is binary.")
        else:
            self.aux_adj = self.adjacency
            self.method = "binary"

        if (sorting_method == "default") and (self.is_weighted):
            sorting_method = "random"
        elif (sorting_method == "default") and (not self.is_weighted):
            sorting_method = "random"

        sort_func = {
                     "random": lambda x: AX.shuffled_edges(x, True),
                     "degrees": None,
                     "strengths": None,
                    }

        try:
            self.sorting_function = sort_func[sorting_method]
        except:
            raise ValueError("Sorting method can be 'random', 'degrees' or 'strengths'.")

        surp_fun = {
                    "binary": lambda x, y: CD.calculate_surprise_logsum_clust_bin(x, y, True),
                    "weighted": lambda x, y: CD.calculate_surprise_logsum_clust_weigh(x, y, True),
                    }

        try:
            self.surprise_function = surp_fun[self.method]
        except:
            raise ValueError("Comunity detection method can be 'binary' or 'weighted'.")

        self.partition_labeler = lambda x: CD.labeling_communities(x)

    def _set_initial_guess_cd(self,
                              initial_guess):
        if initial_guess is None:
            self.init_guess = np.array([k for k in range(self.n_nodes)])
        elif isinstance(initial_guess, np.ndarray):
            self.init_guess = initial_guess
        elif isinstance(initial_guess, list):
            self.init_guess = np.array(initial_guess)

        if self.init_guess.shape[0] != self.n_nodes:
            raise ValueError("The length of the initial guess provided is different from the network number of nodes.")

    def _set_solved_problem(self, sol):
        self.solution = sol[0]
        self.log_surprise = sol[1]
        self.surprise = np.exp(-self.log_surprise)