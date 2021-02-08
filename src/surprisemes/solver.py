import numpy as np


def solver_cp(adjacency_matrix,
              cluster_assignment,
              num_sim,
              sort_edges,
              calculate_surprise,
              correct_partition_labeling,
              flipping_function,
              print_output=False):
    """[summary]

    :param adjacency_matrix: [description]
    :type adjacency_matrix: [type]
    :param cluster_assignment: [description]
    :type cluster_assignment: [type]
    :param num_sim: [description]
    :type num_sim: [type]
    :param sort_edges: [description]
    :type sort_edges: [type]
    :param calculate_surprise: [description]
    :type calculate_surprise: [type]
    :param correct_partition_labeling: [description]
    :type correct_partition_labeling: [type]
    :param print_output: [description], defaults to False
    :type print_output: bool, optional
    :return: [description]
    :rtype: [type]
    """
    surprise = 0
    edges_sorted = sort_edges(adjacency_matrix)
    sim = 0
    while(sim < num_sim):
        # edges_counter = 0
        for [u, v] in edges_sorted:
            # surprise_old = surprise
            cluster_assignment_temp1 = cluster_assignment.copy()
            cluster_assignment_temp2 = cluster_assignment.copy()

            if cluster_assignment[u] != cluster_assignment[v]:
                cluster_assignment_temp1[v] = cluster_assignment[u]
                cluster_assignment_temp2[u] = cluster_assignment[v]

                surprise_temp1 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp1)
                if (surprise_temp1 >= surprise):
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1

                surprise_temp2 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp2)
                if (surprise_temp2 >= surprise):
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2

            else:  # cluster_assignment is the same
                cluster_assignment_temp1[v] = 1 - cluster_assignment[v]
                cluster_assignment_temp2[u] = 1 - cluster_assignment[u]

                surprise_temp1 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp1)
                if (surprise_temp1 >= surprise):
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1

                surprise_temp2 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp2)
                if (surprise_temp2 >= surprise):
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2

        if print_output:
            print()
            print(surprise)
        sim += 1

    if len(cluster_assignment) <= 500:
        n_flips = 100
    else:
        n_flips = int(len(cluster_assignment) * 0.2)

    flips = 0
    while(flips < n_flips):
        cluster_assignment_temp = flipping_function(cluster_assignment.copy)
        surprise_temp = calculate_surprise(adjacency_matrix,
                                           cluster_assignment_temp)
        if (surprise_temp > surprise):
            cluster_assignment = cluster_assignment_temp.copy()
            surprise = surprise_temp
        flips += 1

    cluster_assignment = correct_partition_labeling(adjacency_matrix,
                                                    cluster_assignment)
    return cluster_assignment, surprise


def solver_com_det(adjacency_matrix,
                   cluster_assignment,
                   num_sim,
                   sort_edges,
                   calculate_surprise,
                   correct_partition_labeling,
                   prob_mix,
                   flipping_function,
                   print_output=False):
    """[summary]

    :param adjacency_matrix: [description]
    :type adjacency_matrix: [type]
    :param cluster_assignment: [description]
    :type cluster_assignment: [type]
    :param num_sim: [description]
    :type num_sim: [type]
    :param sort_edges: [description]
    :type sort_edges: [type]
    :param calculate_surprise: [description]
    :type calculate_surprise: [type]
    :param correct_partition_labeling: [description]
    :type correct_partition_labeling: [type]
    :param prob_mix: [description]
    :type prob_mix: [type]
    :param print_output: [description], defaults to False
    :type print_output: bool, optional
    :return: [description]
    :rtype: [type]
    """
    prob_random = (1-prob_mix)/2
    surprise = 0

    contatore_break = 0

    sim = 0
    while(sim < num_sim):
        previous_surprise = surprise
        E_sorted = sort_edges(adjacency_matrix)
        print(E_sorted)
        for [u, v] in E_sorted:
            if cluster_assignment[u] != cluster_assignment[v]:
                cluster_assignement_temp = cluster_assignment.copy()
                random_number = np.random.uniform()
                if (random_number > prob_mix) & (random_number <= (prob_mix+prob_random)):
                    cluster_assignement_temp[v] = cluster_assignment[u]
                    temp_surprise = calculate_surprise(adjacency_matrix,
                                                       cluster_assignement_temp)

                elif (random_number > (prob_mix+prob_random)):
                    cluster_assignement_temp[u] = cluster_assignment[v]
                    temp_surprise = calculate_surprise(adjacency_matrix,
                                                       cluster_assignement_temp)
                else:
                    aux_cluster = cluster_assignement_temp[u]
                    cluster_assignement_temp[cluster_assignement_temp == aux_cluster] = cluster_assignement_temp[v]
                    temp_surprise = calculate_surprise(adjacency_matrix,
                                                       cluster_assignement_temp)

                if temp_surprise > surprise:
                    cluster_assignment = cluster_assignement_temp.copy()
                    surprise = temp_surprise

        if surprise > previous_surprise:
            contatore_break = 0
        else:
            contatore_break += 1

        if contatore_break >= 10:
            break
        if print_output:
            print(surprise)
        sim += 1

    if len(cluster_assignment) <= 500:
        n_flips = 100
    else:
        n_flips = int(len(cluster_assignment) * 0.2)

    flips = 0
    while(flips < n_flips):
        cluster_assignment_temp = flipping_function(cluster_assignment.copy)
        surprise_temp = calculate_surprise(adjacency_matrix,
                                           cluster_assignment_temp)
        if (surprise_temp > surprise):
            cluster_assignment = cluster_assignment_temp.copy()
            surprise = surprise_temp
        flips += 1

    cluster_assignement_proper = correct_partition_labeling(cluster_assignment.copy())
    return cluster_assignement_proper, surprise
