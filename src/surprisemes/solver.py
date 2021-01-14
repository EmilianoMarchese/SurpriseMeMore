import numpy as np


def solver_cp(adjacency_matrix,
              num_sim,
              sort_edges,
              calculate_surprise,
              correct_partition_labeling,
              is_directed=True,
              print_output=False):

    surprise = 0
    cluster_assignment = np.ones(len(adjacency_matrix), dtype=int)

    edges_sorted = sort_edges(adjacency_matrix)
    sim = 0
    while(sim < num_sim):
        edges_counter = 0
        for [u, v] in edges_sorted:
            # surprise_old = surprise
            cluster_assignment_temp1 = cluster_assignment.copy()
            cluster_assignment_temp2 = cluster_assignment.copy()

            if cluster_assignment[u] != cluster_assignment[v]:
                cluster_assignment_temp1[v] = cluster_assignment[u]
                cluster_assignment_temp2[u] = cluster_assignment[v]

                surprise_temp1 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp1,
                                                    is_directed)
                if (surprise_temp1 >= surprise):
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1

                surprise_temp2 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp2,
                                                    is_directed)
                if (surprise_temp2 >= surprise):
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2

            else:  # cluster_assignment is the same
                cluster_assignment_temp1[v] = 1 - cluster_assignment[v]
                cluster_assignment_temp2[u] = 1 - cluster_assignment[u]

                surprise_temp1 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp1,
                                                    is_directed)
                if (surprise_temp1 >= surprise):
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1

                surprise_temp2 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp2,
                                                    is_directed)
                if (surprise_temp2 >= surprise):
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2

            edges_counter += 1
            print(edges_counter, end='\r')
            #if surprise == 0:
            #    print('zero surprise reached')
            #    break
        if print_output:
            print()
            print(surprise)
        sim += 1

    cluster_assignment = correct_partition_labeling(adjacency_matrix,
                                                    cluster_assignment)
    return cluster_assignment, surprise
