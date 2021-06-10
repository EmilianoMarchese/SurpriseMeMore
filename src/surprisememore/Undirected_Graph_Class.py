import numpy as np
from scipy import sparse

from . import auxiliary_function as ax
from . import comdet_functions as cd
from . import cp_functions as cp
from . import solver


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
        self.degree_sequence = None
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
            ) and not sparse.isspmatrix(adjacency):
                raise TypeError(
                    "The adjacency matrix must be passed as a list or numpy"
                    " array or scipy sparse matrix."
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
                    self.adjacency = ax.from_edgelist(edgelist,
                                                      self.is_sparse,
                                                      False)
                    self.edgelist = edgelist
                elif len(edgelist[0]) == 3:
                    self.adjacency = ax.from_weighted_edgelist(edgelist,
                                                               self.is_sparse,
                                                               False)
                    self.edgelist = edgelist
                else:
                    raise ValueError(
                        "This is not an edgelist. An edgelist must be a list"
                        " or array of couples of nodes with optional weights."
                        " Is this an adjacency matrix?"
                    )
        else:
            raise TypeError(
                "UndirectedGraph is missing one positional argument"
                " adjacency.")

        ax.check_adjacency(self.adjacency, self.is_sparse, False)
        if np.sum(self.adjacency) == np.sum(self.adjacency > 0):
            self.degree_sequence = ax.compute_degree(self.adjacency,
                                                     False).astype(np.int64)
        else:
            self.degree_sequence = ax.compute_degree(self.adjacency,
                                                     False).astype(np.int64)
            self.strength_sequence = ax.compute_strength(
                self.adjacency,
                False,
            ).astype(np.float64)
            self.adjacency_weighted = self.adjacency
            self.adjacency = (self.adjacency_weighted.astype(bool)).astype(
                np.int16)
            self.is_weighted = True
        self.n_nodes = len(self.degree_sequence)
        self.n_edges = int(np.sum(self.degree_sequence) / 2)
        self.is_initialized = True

    def set_adjacency_matrix(self, adjacency):
        if self.is_initialized:
            raise ValueError(
                "Graph already contains edges or has a degree sequence."
                " Use 'clean_edges()' first."
            )
        else:
            self._initialize_graph(adjacency=adjacency)

    def set_edgelist(self, edgelist):
        if self.is_initialized:
            raise ValueError(
                "Graph already contains edges or has a degree sequence."
                " Use 'clean_edges()' first."
            )
        else:
            self._initialize_graph(edgelist=edgelist)

    def clean_edges(self):
        self.adjacency = None
        self.edgelist = None
        self.is_initialized = False

    def run_enhanced_cp_detection(self,
                                  initial_guess="ranked",
                                  num_sim=2,
                                  sorting_method="default",
                                  print_output=False):

        self._initialize_problem_cp(
            initial_guess=initial_guess,
            enhanced=True,
            weighted=True,
            sorting_method=sorting_method)

        sol = solver.solver_cp(
            adjacency_matrix=self.aux_adj,
            cluster_assignment=self.init_guess,
            num_sim=num_sim,
            sort_edges=self.sorting_function,
            calculate_surprise=self.surprise_function,
            correct_partition_labeling=self.partition_labeler,
            flipping_function=self.flipping_function,
            print_output=print_output)

        self._set_solved_problem(sol)

    def run_discrete_cp_detection(
            self,
            initial_guess="ranked",
            weighted=None,
            num_sim=2,
            sorting_method="default",
            print_output=False):

        self._initialize_problem_cp(
            initial_guess=initial_guess,
            enhanced=False,
            weighted=weighted,
            sorting_method=sorting_method)

        sol = solver.solver_cp(
            adjacency_matrix=self.aux_adj,
            cluster_assignment=self.init_guess,
            num_sim=num_sim,
            sort_edges=self.sorting_function,
            calculate_surprise=self.surprise_function,
            correct_partition_labeling=self.partition_labeler,
            flipping_function=self.flipping_function,
            print_output=print_output)

        self._set_solved_problem(sol)

    def _initialize_problem_cp(self,
                               initial_guess,
                               enhanced,
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
            if enhanced:
                self.method = "enhanced"
            else:
                self.method = "weighted"

            if hasattr(self, "adjacency_weighted"):
                self.aux_adj = self.adjacency_weighted
                cond2 = (self.aux_adj.astype(np.int64).sum() !=
                         self.aux_adj.sum())
                if cond2:
                    raise ValueError("The selected method works for discrete "
                                     "weights, but the initialised graph has "
                                     "continuous weights.")
            else:
                raise TypeError(
                    "You choose weighted core peryphery detection but the"
                    " graph you initialised is binary.")
        else:
            self.aux_adj = self.adjacency
            self.method = "binary"

        if (sorting_method == "default") and self.is_weighted:
            sorting_method = "random"
        elif (sorting_method == "default") and (not self.is_weighted):
            sorting_method = "jaccard"

        sort_func = {
            "random": lambda x: ax.shuffled_edges(x, False),
            "jaccard": lambda x: ax.jaccard_sorted_edges(x),
            "strengths": None,
        }

        try:
            self.sorting_function = sort_func[sorting_method]
        except Exception:
            raise ValueError(
                "Sorting method can be 'random', 'jaccard', 'degrees'"
                " or 'stengths'.")

        surp_fun = {
            "binary": lambda x, y: cp.calculate_surprise_logsum_cp_bin(
                x,
                y,
                False),
            "weighted": lambda x, y: cp.calculate_surprise_logsum_cp_weigh(
                x,
                y,
                False),
            "enhanced": lambda x, y: cp.calculate_surprise_logsum_cp_enhanced(
                x,
                y,
                False),
        }

        try:
            self.surprise_function = surp_fun[self.method]
        except Exception:
            raise ValueError("CP method can be 'binary' or 'weighted'.")

        self.flipping_function = lambda x: cp.flipping_function_cp(x, 1)

        self.partition_labeler = lambda x, y: cp.labeling_core_periphery(x, y)

    def _set_initial_guess_cp(self, initial_guess):
        # TODO: Sistemare parte pesata
        if isinstance(initial_guess, str):
            if initial_guess == "random":
                self.init_guess = np.ones(self.n_nodes, dtype=np.int32)
                aux_n = int(np.ceil((5 * self.n_nodes) / 100))
                self.init_guess[:aux_n] = 0
                np.random.shuffle(self.init_guess[:aux_n])
            elif initial_guess == "ranked":
                self.init_guess = np.ones(self.n_nodes, dtype=np.int32)
                aux_n = int(np.ceil((5 * self.n_nodes) / 100))
                if self.is_weighted:
                    self.init_guess[
                        self.strength_sequence.argsort()[-aux_n:]] = 0
                else:
                    self.init_guess[
                        self.degree_sequence.argsort()[-aux_n:]] = 0
            elif initial_guess == "eigenvector":
                self.init_guess = ax.eigenvector_init_guess(
                    self.adjacency,
                    False)
            else:
                raise ValueError("Valid values of initial guess are 'random', "
                                 "'eigenvector', 'ranked, or a custom initial"
                                 " guess (np.ndarray or list).")

        elif isinstance(initial_guess, np.ndarray):
            self.init_guess = initial_guess
        elif isinstance(initial_guess, list):
            self.init_guess = np.array(initial_guess)

        if np.unique(self.init_guess).shape[0] != 2:
            raise ValueError("The custom initial_guess passed is not valid."
                             " The initial guess for core-periphery detection"
                             " must have nodes' membership that are 0 or 1."
                             " Pay attention that at least one node has to "
                             "belong to the core (0) or the periphery (1).")

        if self.init_guess.shape[0] != self.n_nodes:
            raise ValueError(
                "The length of the initial guess provided is different from"
                " the network number of nodes.")

    def run_continuous_community_detection(self,
                                           method="aglomerative",
                                           initial_guess="random",
                                           approx=None,
                                           num_sim=2,
                                           num_clusters=None,
                                           prob_mix=0.1,
                                           sorting_method="default",
                                           print_output=False
                                           ):
        self._initialize_problem_cd(
            method=method,
            num_clusters=num_clusters,
            initial_guess=initial_guess,
            enhanced=False,
            weighted=True,
            continuous=True,
            sorting_method=sorting_method)

        if method == "aglomerative":
            sol = solver.solver_com_det_aglom(
                adjacency_matrix=self.aux_adj,
                cluster_assignment=self.init_guess,
                num_sim=num_sim,
                sort_edges=self.sorting_function,
                calculate_surprise=self.surprise_function,
                correct_partition_labeling=self.partition_labeler,
                prob_mix=prob_mix,
                flipping_function=cd.flipping_function_comdet_agl_new,
                approx=approx,
                is_directed=False,
                print_output=print_output)
        elif method == "fixed-clusters":
            sol = solver.solver_com_det_divis(
                adjacency_matrix=self.aux_adj,
                cluster_assignment=self.init_guess,
                num_sim=num_sim,
                sort_edges=self.sorting_function,
                calculate_surprise=self.surprise_function,
                correct_partition_labeling=self.partition_labeler,
                flipping_function=cd.flipping_function_comdet_div_new,
                approx=approx,
                is_directed=False,
                print_output=print_output)
        else:
            raise ValueError("Method can be 'aglomerative' or 'fixed-clusters'.")

        self._set_solved_problem(sol)

    def run_enhanced_community_detection(self,
                                         method="aglomerative",
                                         initial_guess="random",
                                         num_sim=2,
                                         num_clusters=None,
                                         prob_mix=0.1,
                                         sorting_method="default",
                                         print_output=False
                                         ):

        self._initialize_problem_cd(
            method=method,
            num_clusters=num_clusters,
            initial_guess=initial_guess,
            enhanced=True,
            weighted=True,
            continuous=False,
            sorting_method=sorting_method)

        if method == "aglomerative":
            sol = solver.solver_com_det_aglom(
                adjacency_matrix=self.aux_adj,
                cluster_assignment=self.init_guess,
                num_sim=num_sim,
                sort_edges=self.sorting_function,
                calculate_surprise=self.surprise_function,
                correct_partition_labeling=self.partition_labeler,
                prob_mix=prob_mix,
                flipping_function=cd.flipping_function_comdet_agl_new,
                approx=None,
                is_directed=False,
                print_output=print_output)
        elif method == "fixed-clusters":
            sol = solver.solver_com_det_divis(
                adjacency_matrix=self.aux_adj,
                cluster_assignment=self.init_guess,
                num_sim=num_sim,
                sort_edges=self.sorting_function,
                calculate_surprise=self.surprise_function,
                correct_partition_labeling=self.partition_labeler,
                flipping_function=cd.flipping_function_comdet_div_new,
                approx=None,
                is_directed=False,
                print_output=print_output)
        else:
            raise ValueError("Method can be 'aglomerative' or 'fixed-clusters'.")

        self._set_solved_problem(sol)

    def run_discrete_community_detection(self,
                                         method="aglomerative",
                                         initial_guess="random",
                                         weighted=None,
                                         num_sim=2,
                                         num_clusters=None,
                                         prob_mix=0.1,
                                         sorting_method="default",
                                         print_output=False):

        self._initialize_problem_cd(
            method=method,
            num_clusters=num_clusters,
            initial_guess=initial_guess,
            enhanced=False,
            weighted=weighted,
            continuous=False,
            sorting_method=sorting_method)

        if method == "aglomerative":
            sol = solver.solver_com_det_aglom(
                adjacency_matrix=self.aux_adj,
                cluster_assignment=self.init_guess,
                num_sim=num_sim,
                sort_edges=self.sorting_function,
                calculate_surprise=self.surprise_function,
                correct_partition_labeling=self.partition_labeler,
                prob_mix=prob_mix,
                flipping_function=cd.flipping_function_comdet_agl_new,
                approx=None,
                is_directed=False,
                print_output=print_output)
        elif method == "fixed-clusters":
            sol = solver.solver_com_det_divis(
                adjacency_matrix=self.aux_adj,
                cluster_assignment=self.init_guess,
                num_sim=num_sim,
                sort_edges=self.sorting_function,
                calculate_surprise=self.surprise_function,
                correct_partition_labeling=self.partition_labeler,
                flipping_function=cd.flipping_function_comdet_div_new,
                approx=None,
                is_directed=False,
                print_output=print_output)
        else:
            raise ValueError("Method can be 'aglomerative' or 'fixed-clusters'.")

        self._set_solved_problem(sol)

    def _initialize_problem_cd(self,
                               method,
                               num_clusters,
                               initial_guess,
                               enhanced,
                               weighted,
                               continuous,
                               sorting_method):

        self._set_initial_guess_cd(method, num_clusters, initial_guess)
        if weighted is None:
            if self.is_weighted:
                self.aux_adj = self.adjacency_weighted
                self.method = "weighted"
            else:
                self.aux_adj = self.adjacency
                self.method = "binary"
        elif weighted:
            if enhanced:
                self.method = "enhanced"
            elif continuous:
                self.method = "continuous"
            else:
                self.method = "weighted"

            if hasattr(self, "adjacency_weighted"):
                self.aux_adj = self.adjacency_weighted
                cond1 = (self.method == "enhanced" or
                         self.method == "weighted")
                cond2 = (self.aux_adj.astype(np.int64).sum() !=
                         self.aux_adj.sum())
                if cond1 and cond2:
                    raise ValueError("The selected method works for discrete "
                                     "weights, but the initialised graph has "
                                     "continuous weights.")
            else:
                raise TypeError(
                    "You choose weighted core peryphery detection but the"
                    " graph you initialised is binary.")
        else:
            self.aux_adj = self.adjacency
            self.method = "binary"

        if (sorting_method == "default") and self.is_weighted:
            sorting_method = "random"
        elif (sorting_method == "default") and (not self.is_weighted):
            sorting_method = "jaccard"

        sort_func = {
            "random": lambda x: ax.shuffled_edges(x, False),
            "jaccard": lambda x: ax.jaccard_sorted_edges(x),
            "strengths": None,
        }

        try:
            self.sorting_function = sort_func[sorting_method]
        except Exception:
            raise ValueError("Sorting method can be 'random',"
                             " 'strengths' or 'jaccard'.")

        surp_fun = {
            "binary": cd.calculate_surprise_logsum_clust_bin_new,
            "weighted": cd.calculate_surprise_logsum_clust_weigh_new,
            "enhanced": cd.calculate_surprise_logsum_clust_enhanced_new,
            "continuous": cd.calculate_surprise_logsum_clust_weigh_continuos,
        }

        self.surprise_function = surp_fun[self.method]

        # self.flipping_function = lambda x: CD.flipping_function_comdet(x)
        # self.flipping_function = cd.flipping_function_comdet_new

        self.partition_labeler = lambda x: cd.labeling_communities(x)

    def _set_initial_guess_cd(self,
                              method,
                              num_clusters,
                              initial_guess):
        if num_clusters is None and method == "fixed-clusters":
            raise ValueError("When 'fixed-clusters' is passed as clustering"
                             " 'method'"
                             " the 'num_clusters' argument must be specified.")

        if isinstance(initial_guess, str):
            if initial_guess == "random":
                if method == "aglomerative":
                    self.init_guess = np.array(
                        [k for k in np.arange(self.n_nodes, dtype=np.int32)])
                elif method == "fixed-clusters":
                    self.init_guess = np.random.randint(
                        low=num_clusters,
                        size=self.n_nodes)
            elif (initial_guess == "common-neigh-weak") or \
                    (initial_guess == "common-neighbours"):
                if method == "aglomerative":
                    self.init_guess = ax.common_neigh_init_guess_weak(
                        self.adjacency)
                elif method == "fixed-clusters":
                    self.init_guess = ax.fixed_clusters_init_guess_cn(
                        adjacency=self.adjacency,
                        n_clust=num_clusters)
            elif initial_guess == "common-neigh-strong":
                if method == "aglomerative":
                    self.init_guess = ax.common_neigh_init_guess_strong(
                        self.adjacency)
                elif method == "fixed-clusters":
                    self.init_guess = ax.fixed_clusters_init_guess_cn(
                        adjacency=self.adjacency,
                        n_clust=num_clusters)
            else:
                raise ValueError(
                    "The 'initial_guess' selected is not a valid."
                    "Initial guess can be an array specifying nodes membership"
                    " or an initialisation method ['common-neighbours',"
                    " 'random', 'common-neigh-weak', 'common-neigh-strong']."
                    " For more details see documentation.")

        elif isinstance(initial_guess, np.ndarray):
            self.init_guess = initial_guess.astype(np.int32)
        elif isinstance(initial_guess, list):
            self.init_guess = np.array(initial_guess).astype(np.int32)

        if self.init_guess.shape[0] != self.n_nodes:
            raise ValueError(
                "The length of the initial guess provided is different from"
                " the network number of nodes.")

        if (method == "fixed-clusters" and
                np.unique(self.init_guess).shape[0] != num_clusters):
            raise ValueError("The number of clusters of a custom initial guess"
                             " must coincide with 'num_clusters' when the "
                             " fixed-clusters method is applied.")

    def _set_solved_problem(self, sol):
        self.solution = sol[0]
        self.log_surprise = sol[1]
        self.surprise = 10 ** (-self.log_surprise)
