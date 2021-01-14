"Test Undirected Graph Class"

import surprisemes as UG
import numpy as np
import pytest

adjacency = [[0, 3, 3],
             [3, 0, 0],
             [3, 0, 0]]

edgelist = [[1, 2, 3], [1, 3, 3]]

degree_seq = [2, 1, 1]
strength_seq = [6, 3, 3]
n_nodes = 3
n_edges = 2

negative_adj = [[0, 3, -3],
                [3, 0, 0],
                [-3, 0, 0]]

negative_edgelist = [[1, 2, -3], [1, 3, 3]]

asym_adjacency = [[0, 3, 3],
                  [3, 0, 0],
                  [0, 0, 0]]


class TestUndirectedGraphClass():
    def test_initialize_adjacency(self):
        graph = UG.UndirectedGraph(adjacency=adjacency)
        assert(np.all(graph.adjacency == adjacency))
        assert(np.all(graph.dseq == degree_seq))
        assert(np.all(graph.strength_sequence == strength_seq))
        assert(graph.n_nodes == n_nodes)
        assert(graph.n_edges == n_edges)

    def test_initialize_edgelist(self):
        graph = UG.UndirectedGraph(edgelist=edgelist)
        assert(np.all(graph.adjacency == adjacency))
        assert(np.all(graph.dseq == degree_seq))
        assert(np.all(graph.strength_sequence == strength_seq))
        assert(graph.n_nodes == n_nodes)
        assert(graph.n_edges == n_edges)

    def test_negative_entries_adjacency(self):
        with pytest.raises(Exception) as e_info:
            UG.UndirectedGraph(adjacency=negative_adj)
        msg = ("The adjacency matrix entries must be positive.")
        assert e_info.value.args[0] == msg

        with pytest.raises(Exception) as e_info:
            UG.UndirectedGraph(edgelist=negative_edgelist)
        assert e_info.value.args[0] == msg

    def test_wrong_initisialisaton(self):
        with pytest.raises(Exception) as e_info:
            UG.UndirectedGraph(adjacency=negative_edgelist)
        msg = ("Adjacency matrix must be square. If you are passing an edgelist use the positional argument 'edgelist='.")
        assert e_info.value.args[0] == msg

    def test_asymmetric_matrix(self):
        with pytest.raises(Exception) as e_info:
            UG.UndirectedGraph(adjacency=asym_adjacency)
        msg = ("The adjacency matrix seems to be not symmetric, we suggest to use 'DirectedGraphClass'.")
        assert e_info.value.args[0] == msg

    def test_set_functions(self):
        graph = UG.UndirectedGraph(adjacency=adjacency)
        with pytest.raises(Exception) as e_info:
            graph.set_adjacency_matrix(adjacency=adjacency)
        msg = ("Graph already contains edges or has a degree sequence. Use 'clean_edges()' first.")
        assert e_info.value.args[0] == msg

        with pytest.raises(Exception) as e_info:
            graph.set_edgelist(edgelist=edgelist)
        assert e_info.value.args[0] == msg

        graph.clean_edges()
        graph.set_adjacency_matrix(adjacency=adjacency)
        assert(np.all(graph.adjacency == adjacency))
        assert(np.all(graph.dseq == degree_seq))
        assert(np.all(graph.strength_sequence == strength_seq))
        assert(graph.n_nodes == n_nodes)
        assert(graph.n_edges == n_edges)

        graph.clean_edges()
        graph.set_edgelist(edgelist=edgelist)
        assert(np.all(graph.adjacency == adjacency))
        assert(np.all(graph.dseq == degree_seq))
        assert(np.all(graph.strength_sequence == strength_seq))
        assert(graph.n_nodes == n_nodes)
        assert(graph.n_edges == n_edges)
