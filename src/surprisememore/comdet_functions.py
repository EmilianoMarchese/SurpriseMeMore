from . import auxiliary_function as ax

import numpy as np
from mpmath import mp, ncdf, log10, quad, beta, mpf, sqrt, pi
from numba import jit

mp.dps = 100
mp.pretty = True


def calculate_possible_intracluster_links(partitions, is_directed):
    """Computes the number of possible links, given nodes memberships.

    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Total number of possible edges.
    :rtype: float
    """
    _, counts = np.unique(partitions, return_counts=True)
    nr_poss_intr_clust_links = np.sum(counts * (counts - 1))
    if is_directed:
        return nr_poss_intr_clust_links
    else:
        return nr_poss_intr_clust_links / 2


@jit(nopython=True)
def calculate_possible_intracluster_links_new(partitions, is_directed):
    """Computes the number of possible links, given nodes memberships.
    Faster implementation compiled in "nopython" mode.

    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Total number of possible edges.
    :rtype: float
    """
    counts = np.bincount(partitions)
    nr_poss_intr_clust_links = np.sum(counts * (counts - 1))
    if is_directed:
        return nr_poss_intr_clust_links
    else:
        return nr_poss_intr_clust_links / 2


@jit(nopython=True)
def intracluster_links_new(adj, clust_labels, partitions):
    """Computes intracluster links or weights. New implementation

    :param adj: Adjacency matrix.
    :type adj: numpy.ndarray
    :param clust_labels: Labels of changed clusters.
    :type clust_labels: numpy.ndarray
    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
    :return: Number of intra-cluster links/weights.                
    :rtype: numpy.ndarray
    """
    # print(clust_labels, clust_labels.shape, partitions)
    nr_intr_clust_links = np.zeros(clust_labels.shape[0])
    for ii, lab in enumerate(clust_labels):
        indices = np.where(partitions == lab)[0]
        nr_intr_clust_links[ii] = intracluster_links_aux(adj, indices)
    return nr_intr_clust_links


@jit(nopython=True)
def intracluster_links_aux(adj, indices):
    """Computes intra-cluster links or weights given nodes indices.

    :param adj: [description]
    :type adj: [type]
    :param indices: [description]
    :type indices: [type]
    :return: [description]
    :rtype: [type]
    """
    n_links = 0
    for ii in indices:
        for jj in indices:
            n_links += adj[ii, jj]
    return n_links


def calculate_surprise_logsum_clust_bin_new(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        clust_labels,
        args,
        approx,
        is_directed):
    """Calculates the logarithm of the surprise given the current partitions
     for a binary network. New faster implementation reducing the number of
      redundant computations.

    :param adjacency_matrix: Binary adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.                                                       
    :type cluster_assignment: numpy.ndarray
    :param mem_intr_link: Intracluster links per cluster
    :type mem_intr_link: np.array
    :param clust_labels: Labels of changed clusters.
    :type clust_labels: np.array
    :param args: (Observed links, nodes number, possible links)
    :type args: (int, int)
    :param approx:
    :type approx:
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.                                                                              
    :rtype: float                                                                                       
    """
    if is_directed:
        # intracluster links
        if len(clust_labels):
            int_links = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, int_links):
                mem_intr_link[0][node_label] = nr_links

        p = np.sum(mem_intr_link[0])
        int_links = int(p)
        # All the possible intracluster links                                                           
        poss_int_links = int(calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed))
        # Observed links                                                                                
        obs_links = int(args[0])
        # Possible link
        poss_links = int(args[2])
    else:
        # intracluster links
        if len(clust_labels):
            int_links = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, int_links):
                mem_intr_link[0][node_label] = nr_links

        p = np.sum(mem_intr_link[0])
        int_links = int(p / 2)
        # All the possible intracluster links                                                           
        poss_int_links = int(calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed))
        # Observed links                                                                                
        obs_links = int(args[0] / 2)
        # Possible links
        poss_links = int(args[2] / 2)

    if int_links == 0:
        return 0, mem_intr_link

    if approx == "gaussian":
        surprise = binary_surp_gauss_approx(poss_links,
                                            poss_int_links,
                                            obs_links,
                                            int_links)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    elif approx == "asymptotic":
        surprise = asymptot_surp_cd_bin_sum(l_o=int_links,
                                            V=poss_links,
                                            L=obs_links,
                                            V_o=poss_int_links)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    else:
        surprise = surprise_logsum_clust_bin(
            poss_links,
            int_links,
            poss_int_links,
            obs_links)

    return surprise, mem_intr_link


def asymptot_surp_cd_bin(l_o, V, L, V_o):
    f = lambda x: integrand_asympt_cd_b(x, V, L, V_o)
    aux = quad(f, [l_o, L], method="gauss-legendre")
    return aux


def asymptot_surp_cd_bin_sum(l_o, V, L, V_o):
    f = lambda x: integrand_asympt_cd_b(x, V, L, V_o)
    aux = 0
    for l_o_loop in range(l_o, L):
        aux += f(l_o_loop)
    return aux


def integrand_asympt_cd_b(l_o, V, L, V_o):
    p = mpf(L / V)
    p_d = mpf(l_o / V_o)
    p_c = mpf((L - l_o) / (V - V_o))
    a = a_l_o(l_o, V, L, V_o, p, p_d, p_c)
    aux = (a * bernoulli(V, L, p)) / (
                bernoulli(V_o, l_o, p_d) * bernoulli(V - V_o, L - l_o, p_c))
    return aux


def bernoulli(x, y, z):
    aux = (mpf(z) ** y) * (mpf(1 - z) ** (x - y))
    return aux


def a_l_o(l_o, V, L, V_o, p, p_d, p_c):
    aux1 = mpf(V*p*(1-p))**2
    aux2 = mpf(2) * pi * (mpf(V_o * p_d * (1-p_d))**2) * (mpf((V-V_o) * p_c * (1-p_c))**2)
    if aux2:
        aux = sqrt(aux1/aux2)
    else:
        aux = 1
    return aux


def binary_surp_gauss_approx(V, V0, L, l0):
    pi = V0 / V
    P = ncdf(-(l0 - 1 - L * pi) / (np.sqrt(L * pi * (1 - pi))))
    return P


@jit(nopython=True)
def surprise_logsum_clust_bin(F, p, M, m):
    """[summary]

    :param F: [description]
    :type F: [type]
    :param p: [description]
    :type p: [type]
    :param M: [description]
    :type M: [type]
    :param m: [description]
    :type m: [type]
    :return: [description]
    :rtype: [type]
    """
    # stop = False
    min_p = min(M, m)

    logP = loghyperprobability(F, p, M, m)
    for p_loop in range(p, min_p + 1):
        if p_loop == p:
            continue
        nextLogP = loghyperprobability(F, p_loop, M, m)
        [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)
        if stop:
            break

    return -logP


@jit(nopython=True)
def loghyperprobability(F, p, M, m):
    """Evaluates logarithmic hypergeometric distribution

    :param F:
    :type F:
    :param p:
    :type p:
    :param M:
    :type M:
    :param m:
    :type m:
    :return:
    :rtype:
    """
    logH = ax.logc(M, p) + ax.logc(F - M, m - p) - ax.logc(F, m)
    return logH


def calculate_surprise_logsum_clust_weigh_new(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        clust_labels,
        args,
        approx,
        is_directed):
    """Calculates the logarithm of the surprise given the current partitions
     for a weighted network. New faster implementation.

    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.
    :type cluster_assignment: numpy.ndarray
    :param mem_intr_link:
    :type mem_intr_link:
    :param clust_labels:
    :type clust_labels:
    :param args:
    :type args:
    :param approx:
    :type approx:
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.
    :rtype: float
    """
    if is_directed:
        # intracluster weights
        if len(clust_labels):
            w = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, w):
                mem_intr_link[1][node_label] = nr_links

        p = np.sum(mem_intr_link[1])
        intr_weights = p
        # intracluster possible links
        poss_intr_links = calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed)
        # Total Weight
        tot_weights = args[1]
        # Possible links
        poss_links = args[2]
        # extracluster links
        inter_links = poss_links - poss_intr_links
    else:
        # intracluster weights
        if len(clust_labels):
            w = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, w):
                mem_intr_link[1][node_label] = nr_links

        p = np.sum(mem_intr_link[1])
        intr_weights = p / 2
        # intracluster possible links
        poss_intr_links = calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed)
        # Total Weight
        tot_weights = args[1] / 2
        # Possible links
        poss_links = args[2] / 2
        # extracluster links
        inter_links = poss_links - poss_intr_links

    if intr_weights == 0:
        return 0, mem_intr_link

    if approx == "gaussian":
        surprise = weighted_suprise_approx(
            poss_links,
            tot_weights,
            intr_weights,
            poss_intr_links)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    elif approx == "asymptotic":
        surprise = asymptot_surp_cd_wei_sum(poss_links,
                                            tot_weights,
                                            poss_intr_links,
                                            intr_weights)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    else:
        surprise = surprise_logsum_clust_weigh(
            poss_intr_links,
            intr_weights,
            inter_links,
            tot_weights,
            poss_links)
    return surprise, mem_intr_link


def asymptot_surp_cd_wei(V, W, V_o, w_o):
    f = lambda x: integrand_asympt_cd_w(x, V, W, V_o)
    aux = quad(f, [w_o, W], method="gauss-legendre")
    return aux


@jit(forceobj=True)
def asymptot_surp_cd_wei_sum(V, W, V_o, w_o):
    aux_surp = 0
    for w_o_loop in range(int(w_o), int(W)):
        aux = integrand_asympt_cd_w(w_o_loop, V, W, V_o)
        aux_surp += aux
        if aux_surp == 0:
            break
        if aux / aux_surp < 1e-3:
            break
    return aux_surp


def integrand_asympt_cd_w(w_o, V, W, V_o):
    q = mpf(W/(V + W -1))
    q_d = mpf(w_o/(V_o + w_o - 1))
    q_c = mpf((W - w_o)/(V - V_o + W - w_o - 1))
    C = C_w_o(V, W, V_o, w_o, q, q_d, q_c)
    aux = (C * geometric(V, W, q))/(geometric(V_o, w_o, q_d)*geometric(V-V_o, W-w_o, q_c))
    return aux


def geometric(x, y, z):
    aux = (mpf(z)**y) * (mpf(1 - z)**x)
    return aux


def C_w_o(V, W, V_o, w_o, q, q_d, q_c):
    aux = sqrt(mpf(V*q)**2 / (mpf(2)*pi*(mpf(V_o*q_d)**2)*(mpf((V-V_o)*q_c)**2)))
    return aux


def weighted_suprise_approx(V, W, w_o, V_o):
    """Gaussian approximation of surprise.
    """
    rho = W / V
    aux = ncdf(-(w_o - 1 - V_o * rho) / (np.sqrt(V_o * rho * (1 + rho))))
    return aux


@jit(nopython=True)
def surprise_logsum_clust_weigh(Vi, w, Ve, W, V):
    """

    :param Vi: [description]
    :type Vi: [type]
    :param w: [description]
    :type w: [type]
    :param Ve: [description]
    :type Ve: [type]
    :param W: [description]
    :type W: [type]
    :param V: [description]
    :type V: [type]
    :return: [description]
    :rtype: [type]
    """
    # stop = False

    logP = lognegativehyperprobability(Vi, w, Ve, W, V)
    for w_loop in range(w, W):
        if w_loop == w:
            continue
        nextLogP = lognegativehyperprobability(Vi, w_loop, Ve, W, V)
        [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)
        if stop:
            break

    return -logP


@jit(nopython=True)
def lognegativehyperprobability(Vi, w, Ve, W, V):
    """Evaluates logarithmic hypergeometric distribution

    :param Vi:
    :type Vi:
    :param w:
    :type w:
    :param Ve:
    :type Ve:
    :param W:
    :type W:
    :param V:
    :type V:
    :return:
    :rtype:
    """
    logh = ax.logc(
        Vi + w - 1,
        w) + ax.logc(
        Ve + W - w,
        W - w) - ax.logc(
        V + W,
        W)
    return logh


@jit(nopython=True)
def intracluster_links_enh_new(adj, clust_labels, partitions):
    """Computes intracluster links and weights for enhanced community
     detection method.

    :param adj: Adjacency matrix.
    :type adj: numpy.array
    :param clust_labels:
    :type clust_labels:
    :param partitions: Nodes memberships.
    :type partitions: numpy.array
    :return: Number of intra-cluster links/weights.
    :rtype: float
    """
    nr_intr_clust_links = np.zeros(clust_labels.shape[0])
    intr_weight = np.zeros(clust_labels.shape[0])
    for ii, lab in enumerate(clust_labels):
        indices = np.where(partitions == lab)[0]
        aux_l, aux_w = intracluster_links_aux_enh(adj, indices)
        nr_intr_clust_links[ii] = aux_l
        intr_weight[ii] = aux_w
    return nr_intr_clust_links, intr_weight


@jit(nopython=True)
def intracluster_links_aux_enh(adj, indices):
    """Computes intra-cluster links or weights given nodes indices.

    :param adj: [description]
    :type adj: [type]
    :param indices: [description]
    :type indices: [type]
    :return: [description]
    :rtype: [type]
    """
    weight = 0.0
    n_links = 0.0
    for ii in indices:
        for jj in indices:
            if adj[ii, jj]:
                weight += adj[ii, jj]
                n_links += 1
    return n_links, weight


def calculate_surprise_logsum_clust_enhanced_new(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        clust_labels,
        args,
        approx,
        is_directed):
    """Calculates, for a weighted network, the logarithm of the enhanced
     surprise given the current partitioning.

    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.
    :type cluster_assignment: numpy.ndarray
    :param mem_intr_link:
    :type mem_intr_link:
    :param clust_labels:
    :type clust_labels:
    :param args:
    :type args:
    :param approx:
    :type approx:
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.
    :rtype: float
    """
    if is_directed:
        # intracluster weights
        if len(clust_labels):
            l_aux, w_aux = intracluster_links_enh_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links, w_int in zip(clust_labels, l_aux, w_aux):
                mem_intr_link[0][node_label] = nr_links
                mem_intr_link[1][node_label] = w_int

        l_o = int(mem_intr_link[0].sum())
        w_o = int(mem_intr_link[1].sum())

        # intracluster possible links
        V_o = int(calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed))
        # Total Weight
        W = int(args[1])
        L = int(args[0])
        # Possible links
        # n = adjacency_matrix.shape[0]
        V = int(args[2])
        # extracluster links
        # inter_links = V - V_o
    else:
        # intracluster weights
        if len(clust_labels):
            l_aux, w_aux = intracluster_links_enh_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links, w_int in zip(clust_labels, l_aux, w_aux):
                mem_intr_link[0][node_label] = nr_links
                mem_intr_link[1][node_label] = w_int

        l_o = int(mem_intr_link[0].sum() / 2)
        w_o = int(mem_intr_link[1].sum() / 2)

        # intracluster possible links
        V_o = int(calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed))
        # Total Weight
        W = int(args[1] / 2)
        L = int(args[0] / 2)
        # Possible links
        # n = adjacency_matrix.shape[0]
        V = int(args[2] / 2)
        # extracluster links
        # inter_links = V - V_o

    if l_o == 0:
        return 0, mem_intr_link

    # print("V_0", V_o, "l_0", l_o, "w_0", w_o, "V", V, "L", L, "W", W)

    surprise = surprise_logsum_clust_enh(V_o, l_o, w_o, V, L, W)
    return surprise, mem_intr_link


@jit(nopython=True)
def surprise_logsum_clust_enh(V_o, l_o, w_o, V, L, W):
    # stop = False
    # stop1 = False
    min_l_loop = min(L, V_o)

    logP = logenhancedhypergeometric(V_o, l_o, w_o, V, L, W)
    logP1 = logP
    w_loop = w_o

    for l_loop in range(l_o, min_l_loop + 1):
        if l_loop == 0:
            continue
        for w_loop in range(w_o - l_loop + l_o, W - L + l_o + 1):
            if w_loop <= 0:
                continue
            if (w_loop == w_o) and (l_loop == l_o):
                continue
            nextLogP = logenhancedhypergeometric(V_o, l_loop, w_loop, V, L, W)
            [logP, stop] = ax.sumLogProbabilities(nextLogP, logP)
            if stop:
                break
        nextLogP1 = logenhancedhypergeometric(V_o, l_loop, w_loop, V, L, W)
        [logP1, stop1] = ax.sumLogProbabilities(nextLogP1, logP1)
        if stop1:
            break

    return -logP1


@jit(nopython=True)
def logenhancedhypergeometric(V_o, l_o, w_o, V, L, W):
    if l_o < L:
        aux1 = (ax.logc(V_o, l_o) + ax.logc(V - V_o, L - l_o)) - ax.logc(V, L)
        aux2 = (ax.logc(w_o - 1, w_o - l_o) + ax.logc(W - w_o - 1, (W - L) - (
                    w_o - l_o))) - ax.logc(W - 1, W - L)
    else:
        aux1 = (ax.logc(V_o, L) - ax.logc(V, L))
        aux2 = (ax.logc(w_o, w_o - L) - ax.logc(W, L))
    return aux1 + aux2


def calculate_surprise_logsum_clust_weigh_continuos(
        adjacency_matrix,
        cluster_assignment,
        mem_intr_link,
        clust_labels,
        args,
        approx,
        is_directed):
    """Calculates the logarithm of the continuos surprise given the current
     partitions for a weighted network. New faster implementation.


    :param adjacency_matrix: Weighted adjacency matrix.
    :type adjacency_matrix: numpy.ndarray
    :param cluster_assignment: Nodes memberships.
    :type cluster_assignment: numpy.ndarray
    :param mem_intr_link:
    :type mem_intr_link:
    :param clust_labels:
    :type clust_labels:
    :param args:
    :type args:
    :param approx:
    :type approx:
    :param is_directed: True if the graph is directed.
    :type is_directed: bool
    :return: Log-surprise.
    :rtype: float
    """
    if is_directed:
        # intracluster weights
        if len(clust_labels):
            w = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, w):
                mem_intr_link[1][node_label] = nr_links

        p = np.sum(mem_intr_link[1])
        intr_weights = p
        # intracluster possible links
        poss_intr_links = calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed)
        # Total Weight
        tot_weights = args[1]
        # Possible links
        poss_links = args[2]
        # extracluster links
        # inter_links = poss_links - poss_intr_links
    else:
        # intracluster weights
        if len(clust_labels):
            w = intracluster_links_new(
                adj=adjacency_matrix,
                clust_labels=clust_labels,
                partitions=cluster_assignment)

            for node_label, nr_links in zip(clust_labels, w):
                mem_intr_link[1][node_label] = nr_links

        p = np.sum(mem_intr_link[1])
        intr_weights = p / 2
        # intracluster possible links
        poss_intr_links = calculate_possible_intracluster_links_new(
            cluster_assignment,
            is_directed)
        # Total Weight
        tot_weights = args[1] / 2
        # Possible links
        poss_links = args[2] / 2
        # extracluster links
        # inter_links = poss_links - poss_intr_links

    if intr_weights == 0:
        return 0, mem_intr_link

    if approx == "gaussian":
        surprise = weighted_suprise_approx(
            poss_links,
            tot_weights,
            intr_weights,
            poss_intr_links)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    elif approx == "asymptotic":
        surprise = asymptot_surp_cd_wei(
            poss_links,
            tot_weights,
            poss_intr_links,
            intr_weights)
        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0
    else:
        surprise = continuous_surprise_clust(
            poss_links,
            tot_weights,
            intr_weights,
            poss_intr_links)

        if surprise > 0:
            surprise = np.float64(-log10(surprise))
        else:
            surprise = 0

    return surprise, mem_intr_link


def continuous_surprise_clust(V, W, w_o, V_o):
    f = lambda x: integrand_clust(x, V, W, V_o)
    aux_surp = quad(f, [w_o, W], method="gauss-legendre")
    return aux_surp


def integrand_clust(w_o, V, W, V_o):
    aux = W * beta(V, W) / (w_o*beta(V_o, w_o) * (W-w_o) * beta(V-V_o, W-w_o))
    return aux


def labeling_communities(partitions):
    """Gives labels to communities from 0 to number of communities minus one.

    :param partitions: Nodes memberships.
    :type partitions: numpy.ndarray
    :return: Re-labeled nodes memberships.
    :rtype: numpy.ndarray
    """
    different_partitions = np.unique(partitions, return_counts=True)
    aux_argsort = np.argsort(different_partitions[1])[::-1]
    ordered_clusters = different_partitions[0][aux_argsort]
    new_partitioning = partitions.copy()
    new_partitions = np.array([k for k in range(len(ordered_clusters))])
    for old_part, new_part in zip(ordered_clusters, new_partitions):
        # print(old_part,new_part)
        indices_old = np.where(partitions == old_part)[0]
        new_partitioning[indices_old] = new_part

    return new_partitioning


def flipping_function_comdet_agl_new(
        calculate_surprise,
        adj,
        membership,
        mem_intr_link,
        args,
        surprise,
        approx,
        is_directed):
    list_neigh = ax.compute_neighbours(adj)

    for node, node_label in zip(np.arange(membership.shape[0]), membership):
        for node2 in list_neigh[node]:
            new_clust = membership[node2]
            if node_label != new_clust:
                aux_membership = membership.copy()
                aux_membership[node] = new_clust
                # print(np.array([node_label, new_clust]))
                temp_surprise, temp_mem_intr_link = calculate_surprise(
                    adjacency_matrix=adj,
                    cluster_assignment=aux_membership,
                    mem_intr_link=mem_intr_link.copy(),
                    clust_labels=np.array([node_label, new_clust]),
                    args=args,
                    approx=approx,
                    is_directed=is_directed)
                if temp_surprise > surprise:
                    membership = aux_membership.copy()
                    surprise = temp_surprise
                    mem_intr_link = temp_mem_intr_link

    return membership


def flipping_function_comdet_div_new(
        calculate_surprise,
        adj,
        membership,
        mem_intr_link,
        args,
        surprise,
        approx,
        is_directed):
    list_neigh = ax.compute_neighbours(adj)

    for node, node_label in zip(np.arange(membership.shape[0]), membership):
        if len(np.where(membership == node_label)[0]) == 1:
            continue
        for node2 in list_neigh[node]:
            new_clust = membership[node2]
            if node_label != new_clust:
                aux_membership = membership.copy()
                aux_membership[node] = new_clust
                # print(np.array([node_label, new_clust]))
                temp_surprise, temp_mem_intr_link = calculate_surprise(
                    adjacency_matrix=adj,
                    cluster_assignment=aux_membership,
                    mem_intr_link=mem_intr_link.copy(),
                    clust_labels=np.array([node_label, new_clust]),
                    args=args,
                    approx=approx,
                    is_directed=is_directed)
                if temp_surprise > surprise:
                    membership = aux_membership.copy()
                    surprise = temp_surprise
                    mem_intr_link = temp_mem_intr_link

    return membership
