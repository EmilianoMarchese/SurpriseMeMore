# Test Initialize Problem instances for the two Graph classes

import surprisemes as surp
import networkx as nx
import numpy as np

np.random.seed(22)

adjacency_und = nx.to_numpy_array(nx.karate_club_graph())

init_guess_un = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

adjacency_und_w = adjacency_und * 5

adjacency_dir = nx.to_numpy_array(nx.karate_club_graph())

for ii in np.arange(adjacency_dir.shape[0]):
    for jj in np.arange(ii, adjacency_dir.shape[0]):
        if np.random.random() > 0.6 and adjacency_dir[ii, jj] > 0:
            if np.random.random() > 0.5:
                adjacency_dir[ii, jj] = 0
            else:
                adjacency_dir[jj, ii] = 0

init_guess_dir = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1])

adjacency_dir_w = adjacency_dir * 5


class TestInitialGuess:
    def test_initial_guess_cp_undir(self):
        graph = surp.UndirectedGraph(adjacency=adjacency_und)

        graph._set_initial_guess_cp(initial_guess=None)

        assert (np.all(graph.init_guess == init_guess_un))

    def test_initial_guess_cp_dir(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        graph._set_initial_guess_cp(initial_guess=None)

        assert (np.all(graph.init_guess == init_guess_dir))

    def test_initial_guess_cp_undir_custom(self):
        graph = surp.UndirectedGraph(adjacency=adjacency_und)

        init_guess = np.random.randint(low=2, size=graph.n_nodes)

        graph._set_initial_guess_cp(initial_guess=init_guess)

        assert (np.all(graph.init_guess == init_guess))

    def test_initial_guess_cp_dir_custom(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        init_guess = np.random.randint(low=2, size=graph.n_nodes)

        graph._set_initial_guess_cp(initial_guess=init_guess)

        assert (np.all(graph.init_guess == init_guess))

    def test_initial_guess_cd_aglom_undir(self):
        graph = surp.UndirectedGraph(adjacency=adjacency_und)

        init_guess = np.array(
                [k for k in np.arange(graph.n_nodes, dtype=np.int32)])

        graph._set_initial_guess_cd(method="aglomerative",
                                    num_clusters=None,
                                    initial_guess=None)

        assert (np.all(graph.init_guess == init_guess))

    def test_initial_guess_cd_aglom_dir(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        init_guess = np.array(
            [k for k in np.arange(graph.n_nodes, dtype=np.int32)])

        graph._set_initial_guess_cd(method="aglomerative",
                                    num_clusters=None,
                                    initial_guess=None)

        assert (np.all(graph.init_guess == init_guess))

    def test_initial_guess_cd_aglom_undir_common_neigh(self):
        graph = surp.UndirectedGraph(adjacency=adjacency_und)

        graph._set_initial_guess_cd(method="aglomerative",
                                    num_clusters=None,
                                    initial_guess=None)

        assert (np.unique(graph.init_guess).shape[0] < graph.n_nodes)

    def test_initial_guess_cd_aglom_dir_common_neigh(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        graph._set_initial_guess_cd(method="aglomerative",
                                    num_clusters=None,
                                    initial_guess=None)

        assert (np.unique(graph.init_guess).shape[0] < graph.n_nodes)

    def test_initial_guess_cd_aglom_undir_custom(self):
        graph = surp.UndirectedGraph(adjacency=adjacency_und)

        init_guess = np.array([k for k in np.arange(graph.n_nodes-10)] +
                              [k for k in np.arange(10)])

        graph._set_initial_guess_cd(method="aglomerative",
                                    num_clusters=None,
                                    initial_guess=init_guess)

        assert (np.all(graph.init_guess == init_guess))

    def test_initial_guess_cd_aglom_dir_custom(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        init_guess = np.array([k for k in np.arange(graph.n_nodes - 10)] +
                              [k for k in np.arange(10)])

        graph._set_initial_guess_cd(method="aglomerative",
                                    num_clusters=None,
                                    initial_guess=init_guess)

        assert (np.all(graph.init_guess == init_guess))

    def test_initial_guess_cd_divis_undir(self):
        graph = surp.UndirectedGraph(adjacency=adjacency_und)

        num_clusters = 3
        graph._set_initial_guess_cd(method="fixed-clusters",
                                    num_clusters=num_clusters,
                                    initial_guess=None)

        assert (np.unique(graph.init_guess).shape[0] == num_clusters)

    def test_initial_guess_cd_divis_dir(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        num_clusters = 3
        graph._set_initial_guess_cd(method="fixed-clusters",
                                    num_clusters=num_clusters,
                                    initial_guess=None)

        assert (np.unique(graph.init_guess).shape[0] == num_clusters)

    def test_initial_guess_cd_divis_undir_common_neigh(self):
        graph = surp.UndirectedGraph(adjacency=adjacency_und)

        num_clusters = 3
        graph._set_initial_guess_cd(method="fixed-clusters",
                                    num_clusters=num_clusters,
                                    initial_guess="common-neighbours")

        assert (np.unique(graph.init_guess).shape[0] == num_clusters)

    def test_initial_guess_cd_divis_dir_common_neigh(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        num_clusters = 3
        graph._set_initial_guess_cd(method="fixed-clusters",
                                    num_clusters=num_clusters,
                                    initial_guess="common-neighbours")

        assert (np.unique(graph.init_guess).shape[0] == num_clusters)

    def test_initial_guess_cd_divis_undir_custom(self):
        graph = surp.UndirectedGraph(adjacency=adjacency_und)

        init_guess = np.array([k for k in np.arange(graph.n_nodes - 10)] +
                              [k for k in np.arange(10)])

        graph._set_initial_guess_cd(method="fixed-clusters",
                                    num_clusters=2,
                                    initial_guess=init_guess)

        assert (np.all(graph.init_guess == init_guess))

    def test_initial_guess_cd_divis_dir_custom(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        init_guess = np.array([k for k in np.arange(graph.n_nodes - 10)] +
                              [k for k in np.arange(10)])

        graph._set_initial_guess_cd(method="fixed-clusters",
                                    num_clusters=2,
                                    initial_guess=init_guess)

        assert (np.all(graph.init_guess == init_guess))

        # TODO: aggiungere i test con initial guess varie per la core periphery


class TestInitializeProblem:
    def test_initialize_problem_undir_cp_1(self):
        graph = surp.DirectedGraph(adjacency=adjacency_dir)

        graph._initialize_problem_cp(initial_guess=None,
                                     weighted=None,
                                     sorting_method="random")

        # TODO: Inserire le funzioni con lambda e fare il paragone

        assert(graph.method == "binary")
        assert(graph.aux_adj == adjacency_dir)
        assert(graph.sorting_function == surp.ax.shuffled_edges)
        assert()





