import numpy as np
from tqdm import tqdm

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
        for [u, v] in tqdm(edges_sorted):
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


def solver_com_det_aglom(
        adjacency_matrix,
        cluster_assignment,
        num_sim,
        sort_edges,
        calculate_surprise,
        correct_partition_labeling,
        prob_mix,
        flipping_function,
        approx,
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
    :param approx:
    :type approx:
    :param is_directed:
    :type is_directed:
    :param print_output: [description], defaults to False
    :type print_output: bool, optional
    :return: [description]
    :rtype: [type]
    """
    prob_random = (1 - prob_mix) / 2

    obs_links = np.sum(adjacency_matrix.astype(bool))
    obs_weights = np.sum(adjacency_matrix)
    n_nodes = int(adjacency_matrix.shape[0])
    poss_links = n_nodes * (n_nodes - 1)
    args = (obs_links, obs_weights, poss_links)

    cluster_assignment = correct_partition_labeling(
        cluster_assignment.copy())
    n_clusters = np.unique(cluster_assignment).shape[0]

    mem_intr_link = np.zeros((2, n_clusters), dtype=np.float64)
    for ii in np.unique(cluster_assignment):
        indices = np.where(cluster_assignment == ii)[0]
        l_aux, w_aux = cd.intracluster_links_aux_enh(
            adjacency_matrix,
            indices)
        mem_intr_link[0][ii] = l_aux
        mem_intr_link[1][ii] = w_aux

    surprise, _ = calculate_surprise(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        np.array([]),
        args,
        approx,
        is_directed)

    contatore_break = 0

    sim = 0
    while sim < num_sim:
        previous_surprise = surprise
        e_sorted = sort_edges(adjacency_matrix)
        # print(cluster_assignment)
        # print(e_sorted)
        for [u, v] in tqdm(e_sorted):
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
                        mem_intr_link.copy(),
                        np.array([clus_u, clus_v]),
                        args,
                        approx,
                        is_directed)
                    # print(cluster_assignement_temp, cluster_assignment)
                    # print("prob_1", u, v, temp_surprise, surprise)
                elif random_number > (prob_mix + prob_random):

                    cluster_assignement_temp[u] = clus_v
                    temp_surprise, temp_mem_intr_link = calculate_surprise(
                        adjacency_matrix,
                        cluster_assignement_temp,
                        mem_intr_link.copy(),
                        np.array([clus_u, clus_v]),
                        args,
                        approx,
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
                        mem_intr_link.copy(),
                        np.array([clus_u, clus_v]),
                        args,
                        approx,
                        is_directed)
                    # print(cluster_assignement_temp, cluster_assignment)
                    # print("mixing", u, v, temp_surprise, surprise)

                if temp_surprise > surprise:
                    cluster_assignment = cluster_assignement_temp.copy()
                    surprise = temp_surprise
                    mem_intr_link = temp_mem_intr_link.copy()

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
        calculate_surprise=calculate_surprise,
        adj=adjacency_matrix,
        membership=cluster_assignment.copy(),
        mem_intr_link=mem_intr_link.copy(),
        args=args,
        surprise=surprise,
        approx=approx,
        is_directed=is_directed)

    cluster_assignement_proper = correct_partition_labeling(
        cluster_assignment.copy())
    return cluster_assignement_proper, surprise


def solver_com_det_divis(
        adjacency_matrix,
        cluster_assignment,
        num_sim,
        sort_edges,
        calculate_surprise,
        correct_partition_labeling,
        flipping_function,
        approx,
        is_directed,
        print_output=False):
    """Community detection solver. It carries out the research of the optimal
    partiton using a greedy strategy with a fixed number of clusters.

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
    :param approx:
    :type approx:
    :param is_directed:
    :type is_directed:
    :param print_output: [description], defaults to False
    :type print_output: bool, optional
    :return: [description]
    :rtype: [type]
    """
    obs_links = np.sum(adjacency_matrix.astype(bool))
    obs_weights = np.sum(adjacency_matrix)
    n_nodes = int(adjacency_matrix.shape[0])
    poss_links = n_nodes * (n_nodes - 1)
    args = (obs_links, obs_weights, poss_links)

    n_clusters = np.unique(cluster_assignment).shape[0]

    mem_intr_link = np.zeros((2, n_clusters), dtype=np.float64)
    for ii in np.unique(cluster_assignment):
        indices = np.where(cluster_assignment == ii)[0]
        l_aux, w_aux = cd.intracluster_links_aux_enh(
            adjacency_matrix,
            indices)
        mem_intr_link[0][ii] = l_aux
        mem_intr_link[1][ii] = w_aux

    surprise, _ = calculate_surprise(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        np.array([]),
        args,
        approx,
        is_directed)

    contatore_break = 0

    sim = 0
    while sim < num_sim:
        previous_surprise = surprise
        e_sorted = sort_edges(adjacency_matrix)
        # print(cluster_assignment)
        # print(E_sorted)
        for [u, v] in tqdm(e_sorted):
            cluster_assignment_temp1 = cluster_assignment.copy()
            cluster_assignment_temp2 = cluster_assignment.copy()

            if cluster_assignment[u] != cluster_assignment[v]:
                clus_u = cluster_assignment[u]
                clus_v = cluster_assignment[v]
                cluster_assignment_temp1[v] = clus_u
                cluster_assignment_temp2[u] = clus_v

                surprise_temp1, temp_mem_intr_link1 = calculate_surprise(
                    adjacency_matrix,
                    cluster_assignment_temp1,
                    mem_intr_link.copy(),
                    np.array([clus_u, clus_v]),
                    args,
                    approx,
                    is_directed)

                aux_n_clus = np.unique(cluster_assignment_temp1).shape[0]
                if (surprise_temp1 > surprise) and (n_clusters == aux_n_clus):
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1
                    mem_intr_link = temp_mem_intr_link1

                surprise_temp2, temp_mem_intr_link2 = calculate_surprise(
                    adjacency_matrix,
                    cluster_assignment_temp2,
                    mem_intr_link.copy(),
                    np.array([clus_u, clus_v]),
                    args,
                    approx,
                    is_directed)

                aux_n_clus = np.unique(cluster_assignment_temp2).shape[0]
                if (surprise_temp2 > surprise) and (n_clusters == aux_n_clus):
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2
                    mem_intr_link = temp_mem_intr_link2

            else:
                clus_u = cluster_assignment[u]
                clus_v = cluster_assignment[v]
                while cluster_assignment_temp1[v] == clus_v and\
                        cluster_assignment_temp2[u] == clus_u:
                    cluster_assignment_temp1[v] = np.random.randint(n_clusters)
                    cluster_assignment_temp2[u] = np.random.randint(n_clusters)

                surprise_temp1, temp_mem_intr_link1 = calculate_surprise(
                    adjacency_matrix,
                    cluster_assignment_temp1,
                    mem_intr_link.copy(),
                    np.array([clus_u, clus_v]),
                    args,
                    approx,
                    is_directed)

                aux_n_clus = np.unique(cluster_assignment_temp1).shape[0]
                if (surprise_temp1 > surprise) and (n_clusters == aux_n_clus):
                    cluster_assignment = cluster_assignment_temp1.copy()
                    surprise = surprise_temp1
                    mem_intr_link = temp_mem_intr_link1

                surprise_temp2, temp_mem_intr_link2 = calculate_surprise(
                    adjacency_matrix,
                    cluster_assignment_temp2,
                    mem_intr_link.copy(),
                    np.array([clus_u, clus_v]),
                    args,
                    approx,
                    is_directed)

                aux_n_clus = np.unique(cluster_assignment_temp2).shape[0]
                if (surprise_temp2 > surprise) and (n_clusters == aux_n_clus):
                    cluster_assignment = cluster_assignment_temp2.copy()
                    surprise = surprise_temp2
                    mem_intr_link = temp_mem_intr_link2

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
        calculate_surprise=calculate_surprise,
        adj=adjacency_matrix,
        membership=cluster_assignment.copy(),
        mem_intr_link=mem_intr_link.copy(),
        args=args,
        surprise=surprise,
        approx=approx,
        is_directed=is_directed)

    cluster_assignement_proper = correct_partition_labeling(
        cluster_assignment.copy())
    return cluster_assignement_proper, surprise
