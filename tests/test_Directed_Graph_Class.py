# Test Directed Graph Class

import surprisemes as dg
import numpy as np
import pytest

adjacency = [[0, 1, 3],
             [3, 0, 0],
             [2, 0, 0]]

edgelist = [(0, 1, 1), (0, 2, 3), (1, 0, 3), (2, 0, 2)]

out_degree = np.array([2, 1, 1])
in_degree = np.array([2, 1, 1])
out_strength = np.array([4, 3, 2])
in_strength = np.array([5, 1, 3])
n_nodes = 3
n_edges = 4

negative_adj = [[0, 1, -3],
                [3, 0, 0],
                [2, 0, 0]]

negative_edgelist = [(0, 1, 1), (0, 2, -3), (1, 0, 3), (2, 0, 2)]


class TestUndirectedGraphClass:
    def test_initialize_adjacency(self):
        graph = dg.DirectedGraph(adjacency=adjacency)

        assert(np.all(graph.adjacency == adjacency))
        assert(np.all(graph.degree_sequence_out == out_degree))
        assert (np.all(graph.degree_sequence_in == in_degree))
        assert(np.all(graph.strength_sequence_out == out_strength))
        assert (np.all(graph.strength_sequence_in == in_strength))
        assert(graph.n_nodes == n_nodes)
        assert(graph.n_edges == n_edges)

    def test_initialize_edgelist(self):
        graph = dg.DirectedGraph(edgelist=edgelist)

        assert (np.all(graph.adjacency == adjacency))
        assert (np.all(graph.degree_sequence_out == out_degree))
        assert (np.all(graph.degree_sequence_in == in_degree))
        assert (np.all(graph.strength_sequence_out == out_strength))
        assert (np.all(graph.strength_sequence_in == in_strength))
        assert (graph.n_nodes == n_nodes)
        assert (graph.n_edges == n_edges)

    def test_negative_entries_adjacency(self):
        with pytest.raises(Exception) as e_info:
            dg.DirectedGraph(adjacency=negative_adj)
        msg = "The adjacency matrix entries must be positive."
        assert e_info.value.args[0] == msg

        with pytest.raises(Exception) as e_info:
            dg.DirectedGraph(edgelist=negative_edgelist)
        assert e_info.value.args[0] == msg

    def test_wrong_initisialisaton(self):
        with pytest.raises(Exception) as e_info:
            dg.DirectedGraph(adjacency=negative_edgelist)
        msg = "Adjacency matrix must be square. If you are passing an" \
              " edgelist use the positional argument 'edgelist='."
        assert e_info.value.args[0] == msg

    def test_set_functions(self):
        graph = dg.DirectedGraph(adjacency=adjacency)
        with pytest.raises(Exception) as e_info:
            graph.set_adjacency_matrix(adjacency=adjacency)
        msg = ("Graph already contains edges or has a degree"
               " sequence. Use 'clean_edges()' first.")
        assert e_info.value.args[0] == msg

        with pytest.raises(Exception) as e_info:
            graph.set_edgelist(edgelist=edgelist)
        assert e_info.value.args[0] == msg

        graph.clean_edges()
        graph.set_adjacency_matrix(adjacency=adjacency)
        assert (np.all(graph.adjacency == adjacency))
        assert (np.all(graph.degree_sequence_out == out_degree))
        assert (np.all(graph.degree_sequence_in == in_degree))
        assert (np.all(graph.strength_sequence_out == out_strength))
        assert (np.all(graph.strength_sequence_in == in_strength))
        assert (graph.n_nodes == n_nodes)
        assert (graph.n_edges == n_edges)

        graph.clean_edges()
        graph.set_edgelist(edgelist=edgelist)
        assert (np.all(graph.adjacency == adjacency))
        assert (np.all(graph.degree_sequence_out == out_degree))
        assert (np.all(graph.degree_sequence_in == in_degree))
        assert (np.all(graph.strength_sequence_out == out_strength))
        assert (np.all(graph.strength_sequence_in == in_strength))
        assert (graph.n_nodes == n_nodes)
        assert (graph.n_edges == n_edges)
