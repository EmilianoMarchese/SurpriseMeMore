import numpy as np

from . import comdet_functions as cd


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
    :param flipping_function:
    :type flipping_function:
    :param print_output: [description], defaults to False
    :type print_output: bool, optional
    :return: [description]
    :rtype: [type]
    """
    surprise = 0
    edges_sorted = sort_edges(adjacency_matrix)
    sim = 0
    while sim < num_sim:
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
                if surprise_temp1 >= surprise:
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1

                surprise_temp2 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp2)
                if surprise_temp2 >= surprise:
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2

            else:  # cluster_assignment is the same
                cluster_assignment_temp1[v] = 1 - cluster_assignment[v]
                cluster_assignment_temp2[u] = 1 - cluster_assignment[u]

                surprise_temp1 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp1)
                if surprise_temp1 >= surprise:
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1

                surprise_temp2 = calculate_surprise(adjacency_matrix,
                                                    cluster_assignment_temp2)
                if surprise_temp2 >= surprise:
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
    while flips < n_flips:
        cluster_assignment_temp = flipping_function(cluster_assignment.copy())
        surprise_temp = calculate_surprise(adjacency_matrix,
                                           cluster_assignment_temp)
        if surprise_temp > surprise:
            cluster_assignment = cluster_assignment_temp.copy()
            surprise = surprise_temp
        flips += 1

    cluster_assignment = correct_partition_labeling(adjacency_matrix,
                                                    cluster_assignment)
    return cluster_assignment, surprise


def solver_com_det(
        adjacency_matrix,
        cluster_assignment,
        num_sim,
        sort_edges,
        calculate_surprise,
        correct_partition_labeling,
        prob_mix,
        flipping_function,
        is_directed,
        print_output=False):
    """Community detection solver. It carries out the research of the optimal
    partiton using a greedy strategy.

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
    :param flipping_function:
    :type flipping_function:
    :param is_directed:
    :type is_directed:
    :param print_output: [description], defaults to False
    :type print_output: bool, optional
    :return: [description]
    :rtype: [type]
    """
    prob_random = (1 - prob_mix) / 2

    obs_links = int(np.sum(adjacency_matrix))
    n_nodes = int(adjacency_matrix.shape[0])
    poss_links = int(n_nodes * (n_nodes - 1))
    args = (obs_links, poss_links)

    mem_intr_link = np.zeros(cluster_assignment.shape[0], dtype=np.int32)
    for ii in np.unique(cluster_assignment):
        indices = np.where(cluster_assignment == ii)[0]
        mem_intr_link[ii] = cd.intracluster_links_aux(
            adjacency_matrix,
            indices)

    surprise, _ = calculate_surprise(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        np.array([]),
        args,
        is_directed)

    contatore_break = 0

    sim = 0
    while sim < num_sim:
        previous_surprise = surprise
        e_sorted = sort_edges(adjacency_matrix)
        # print(cluster_assignment)
        # print(E_sorted)
        for [u, v] in e_sorted:
            if cluster_assignment[u] != cluster_assignment[v]:
                clus_u = cluster_assignment[u]
                clus_v = cluster_assignment[v]
                cluster_assignement_temp = cluster_assignment.copy()
                random_number = np.random.uniform()
                if (random_number > prob_mix) & (
                        random_number <= (prob_mix + prob_random)):
                    cluster_assignement_temp[v] = clus_u
                    temp_surprise, temp_mem_intr_link = calculate_surprise(
                        adjacency_matrix,
                        cluster_assignement_temp,
                        mem_intr_link,
                        np.array([clus_u, clus_v]),
                        args,
                        is_directed)
                    # print(cluster_assignement_temp, cluster_assignment)
                    # print("prob_1", u, v, temp_surprise, surprise)
                elif random_number > (prob_mix + prob_random):

                    cluster_assignement_temp[u] = clus_v
                    temp_surprise, temp_mem_intr_link = calculate_surprise(
                        adjacency_matrix,
                        cluster_assignement_temp,
                        mem_intr_link,
                        np.array([clus_u, clus_v]),
                        args,
                        is_directed)
                    # print(cluster_assignement_temp, cluster_assignment)
                    # print("prob_2", u, v, temp_surprise, surprise)
                else:
                    aux_cluster = cluster_assignement_temp[u]
                    cluster_assignement_temp[
                        cluster_assignement_temp == aux_cluster] = \
                        cluster_assignement_temp[v]
                    temp_surprise, temp_mem_intr_link = calculate_surprise(
                        adjacency_matrix,
                        cluster_assignement_temp,
                        mem_intr_link,
                        np.array([clus_u, clus_v]),
                        args,
                        is_directed)
                    # print(cluster_assignement_temp, cluster_assignment)
                    # print("mixing", u, v, temp_surprise, surprise)

                if temp_surprise > surprise:
                    cluster_assignment = cluster_assignement_temp.copy()
                    surprise = temp_surprise
                    mem_intr_link = temp_mem_intr_link

        if surprise > previous_surprise:
            contatore_break = 0
        else:
            contatore_break += 1

        if contatore_break >= 10:
            break
        if print_output:
            print(surprise)
        sim += 1

    cluster_assignment = flipping_function(
        adjacency_matrix,
        cluster_assignment.copy())

    """"
    if len(cluster_assignment) <= 500:
        n_flips = 100
    else:
        n_flips = int(len(cluster_assignment) * 0.2)

    flips = 0
    while(flips < n_flips):
        cluster_assignment_temp = flipping_function(cluster_assignment.copy())
        surprise_temp = calculate_surprise(adjacency_matrix,
                                           cluster_assignment_temp)
        if (surprise_temp > surprise):
            cluster_assignment = cluster_assignment_temp.copy()
            surprise = surprise_temp
        flips += 1
    """

    cluster_assignement_proper = correct_partition_labeling(
        cluster_assignment.copy())
    return cluster_assignement_proper, surprise


def solver_com_det_old(
        adjacency_matrix,
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
    prob_random = (1 - prob_mix) / 2
    surprise = 0

    contatore_break = 0

    sim = 0
    while (sim < num_sim):
        previous_surprise = surprise
        E_sorted = sort_edges(adjacency_matrix)
        # print(cluster_assignment)
        # print(E_sorted)
        for [u, v] in E_sorted:
            if cluster_assignment[u] != cluster_assignment[v]:
                cluster_assignement_temp = cluster_assignment.copy()
                random_number = np.random.uniform()
                if (random_number > prob_mix) & (
                        random_number <= (prob_mix + prob_random)):
                    cluster_assignement_temp[v] = cluster_assignment[u]
                    temp_surprise = calculate_surprise(adjacency_matrix,
                                                       cluster_assignement_temp)
                    # print(cluster_assignement_temp, cluster_assignment)
                    # print("prob_1", u, v, temp_surprise, surprise)
                elif (random_number > (prob_mix + prob_random)):

                    cluster_assignement_temp[u] = cluster_assignment[v]
                    temp_surprise = calculate_surprise(adjacency_matrix,
                                                       cluster_assignement_temp)
                    # print(cluster_assignement_temp, cluster_assignment)
                    # print("prob_2", u, v, temp_surprise, surprise)
                else:
                    aux_cluster = cluster_assignement_temp[u]
                    cluster_assignement_temp[
                        cluster_assignement_temp == aux_cluster] = \
                        cluster_assignement_temp[v]
                    temp_surprise = calculate_surprise(adjacency_matrix,
                                                       cluster_assignement_temp)
                    # print(cluster_assignement_temp, cluster_assignment)
                    # print("mixing", u, v, temp_surprise, surprise)

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

    cluster_assignment = flipping_function(adjacency_matrix,
                                           cluster_assignment.copy())

    """"
    if len(cluster_assignment) <= 500:
        n_flips = 100
    else:
        n_flips = int(len(cluster_assignment) * 0.2)

    flips = 0
    while(flips < n_flips):
        cluster_assignment_temp = flipping_function(cluster_assignment.copy())
        surprise_temp = calculate_surprise(adjacency_matrix,
                                           cluster_assignment_temp)
        if (surprise_temp > surprise):
            cluster_assignment = cluster_assignment_temp.copy()
            surprise = surprise_temp
        flips += 1
    """

    cluster_assignement_proper = correct_partition_labeling(
        cluster_assignment.copy())
    return cluster_assignement_proper, surprise
