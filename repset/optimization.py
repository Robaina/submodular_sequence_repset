"""
Optimization algorithms
-------------------------
Each returns either a specific subset or an order.
"""

import subprocess
import random
from path import path
import copy
import heapq
import resource
import logging

import numpy
import sklearn.cluster

from repset.similarity import *
from repset.objectives import summaxacross


logger = logging.getLogger('log') # get logger object from main module where parser is located

# random selection
# returns an order
def random_selection(db):
    return random.sample(db.keys(), len(db.keys()))

# naive greedy selecition
# returns an order
def naive_greedy_selection(db, objective, sim):
    not_in_repset = set(db.keys())
    repset = []
    objective_data = objective["base_data"](db, sim)
    for iteration_index in range(len(db.keys())):
        if (iteration_index % 100) == 0: logger.debug("Naive Greedy iteration: %s", iteration_index)
        best_id = None
        best_diff = None
        for seq_id_index, seq_id in enumerate(not_in_repset):
            diff = objective["diff"](db, seq_id, sim, objective_data)
            if (best_diff is None) or (diff > best_diff):
                best_diff = diff
                best_id = seq_id
        repset.append(best_id)
        not_in_repset.remove(best_id)
        objective_data = objective["update"](db, best_id, sim, objective_data)
    return repset

# returns an order
def accelerated_greedy_selection(db, objective, sim, max_evals=float("inf"), diff_approx_ratio=1.0, repset_size=float("inf"), target_obj_val=float("inf")):
    assert diff_approx_ratio <= 1.0
    repset = []
    pq = [(-float("inf"), seq_id) for seq_id in db]
    objective_data = objective["base_data"](db, sim)
    cur_objective = 0
    num_evals = 0
    while (len(repset) < repset_size) and (len(pq) > 1) and (cur_objective < target_obj_val):
        possible_diff, seq_id = heapq.heappop(pq)
        diff = objective["diff"](db, seq_id, sim, objective_data)
        next_diff = -pq[0][0]
        num_evals += 1
        if (num_evals >= max_evals) or (((diff - next_diff) / (abs(diff)+0.01)) >= (diff_approx_ratio - 1.0)):
            repset.append(seq_id)
            objective_data = objective["update"](db, seq_id, sim, objective_data)
            cur_objective += diff
            #assert(abs(cur_objective - objective["eval"](db, repset, sim)) < 1e-3)
            #if (len(repset) % 100) == 0: logger.debug("Accelerated greedy iteration: %s", len(repset))
            #print len(repset), num_evals # XXX
            #print len(repset), cur_objective
            num_evals = 0
        else:
            heapq.heappush(pq, (-diff, seq_id))
    if len(pq) == 1:
        repset.append(pq[0][1])
    return repset

def complement_greedy_selection(db, objective, sim):
    complement_objective = {}
    complement = lambda seq_ids: set(db.keys()) - set(seq_ids)
    complement_objective["eval"] = lambda db, seq_ids, sim: objective["eval"](db, complement(seq_ids), sim)
    complement_objective["diff"] = lambda db, seq_id, sim, data: objective["negdiff"](db, seq_id, sim, data)
    complement_objective["negdiff"] = lambda db, seq_id, sim, data: objective["diff"](db, seq_id, sim, data)
    complement_objective["update"] = lambda db, seq_id, sim, data: objective["negupdate"](db, seq_id, sim, data)
    complement_objective["negupdate"] = lambda db, seq_id, sim, data: objective["update"](db, seq_id, sim, data)
    complement_objective["base_data"] = lambda db, sim: objective["full_data"](db, sim)
    complement_objective["full_data"] = lambda db, sim: objective["base_data"](db, sim)
    repset_order = accelerated_greedy_selection(db, complement_objective, sim)
    return repset_order[::-1]


def stochastic_greedy_selection(db, objective, sim, sample_size, repset_size=float("inf")):
    repset = []
    objective_data = objective["base_data"](db, sim)
    cur_objective = 0
    possible = {seq_id: float("inf") for seq_id in db}
    choosable_seq_ids = set(db.keys())
    sample_size = int(sample_size)
    next_log = 10
    for iter_index in range(min(repset_size, len(db))):
        if iter_index >= next_log:
            if not (logger is None):
                logger.info("stochastic_greedy_selection {} / {}".format(iter_index, min(repset_size, len(db))))
            next_log *= 1.3
        sample = random.sample(choosable_seq_ids, min(sample_size, len(choosable_seq_ids)))
        sample = sorted(sample, key=lambda seq_id: possible[seq_id], reverse=True)
        cur_best_diff = float("-inf")
        cur_best_seq_id = None
        for seq_id in sample:
            if cur_best_diff >= possible[seq_id]:
                break
            diff = objective["diff"](db, seq_id, sim, objective_data)
            possible[seq_id] = diff
            if diff > cur_best_diff:
                cur_best_diff = diff
                cur_best_seq_id = seq_id
        repset.append(cur_best_seq_id)
        choosable_seq_ids.remove(cur_best_seq_id)
        objective_data = objective["update"](db, cur_best_seq_id, sim, objective_data)
        cur_objective += cur_best_diff
    #assert len(repset) == len(db)
    return repset



# Returns a set
# Like graphcdhit_selection, but works for arbitrary objectives
# Uses objective["diff"]
def threshold_selection(db, objective, sim, diff_threshold, order_by_length=True):
    repset = [] # [{"id": id, "objective": objective}]
    objective_data = objective["base_data"](db, sim)
    if order_by_length:
        seq_ids_ordered = sorted(db.keys(), key=lambda seq_id: -len(str(db[seq_id]["seq"])))
    else:
        seq_ids_ordered = random.sample(db.keys(), len(db.keys()))
    next_log = 10
    for iteration_index, seq_id in enumerate(seq_ids_ordered):
        if iteration_index >= next_log:
            if not (logger is None):
                logger.info("threshold_selection {} / {}".format(iteration_index, len(seq_ids_ordered)))
            next_log *= 1.3
        diff = objective["diff"](db, seq_id, sim, objective_data)
        if diff >= diff_threshold:
            repset.append(seq_id)
            objective_data = objective["update"](db, seq_id, sim, objective_data)
    return repset

def nonmonotone_selection(db, objective, sim, modular_bonus):
    id_order = random.sample(db.keys(), len(db.keys()))
    growing_set = set()
    shrinking_set = set(copy.deepcopy(db.keys()))
    growing_set_data = objective["base_data"](db, sim)
    shrinking_set_data = objective["full_data"](db, sim)
    shrinking_set_val = objective["eval"](db, shrinking_set, sim) + modular_bonus * len(shrinking_set)
    growing_set_val = objective["eval"](db, growing_set, sim) + modular_bonus * len(growing_set)
    cur_objective = 0
    for seq_id in id_order:
        growing_imp = objective["diff"](db, seq_id, sim, growing_set_data) + modular_bonus
        shrinking_imp = objective["negdiff"](db, seq_id, sim, shrinking_set_data) - modular_bonus
        norm_growing_imp = max(0, growing_imp)
        norm_shrinking_imp = max(0, shrinking_imp)
        if (norm_growing_imp == 0) and (norm_shrinking_imp == 0): norm_growing_imp, norm_shrinking_imp = 1,1
        if numpy.random.random() < float(norm_growing_imp) / (norm_growing_imp + norm_shrinking_imp):
            growing_set.add(seq_id)
            growing_set_val += growing_imp
            growing_set_data = objective["update"](db, seq_id, sim, growing_set_data)
            #true_growing_set_val = objective["eval"](db, growing_set, sim) + modular_bonus * len(growing_set)
            #if abs(growing_set_val - true_growing_set_val) > 1e-3:
                #logger.error("Miscalculated growing_set_val! calculated: %s ; true: %s", growing_set_val, true_growing_set_val)
        else:
            shrinking_set.remove(seq_id)
            shrinking_set_val += shrinking_imp
            shrinking_set_data = objective["negupdate"](db, seq_id, sim, shrinking_set_data)
            #true_shrinking_set_val = objective["eval"](db, shrinking_set, sim) + modular_bonus * len(shrinking_set)
            #if abs(shrinking_set_val - true_shrinking_set_val) > 1e-3:
                #logger.error("Miscalculated shrinking_set_val! calculated: %s ; true: %s", shrinking_set_val, true_shrinking_set_val)
    return growing_set



# cdhit selection
# seqs: {seq_id: seq}
# Returns a subset
def cdhit_selection(db, workdir, c=0.9):
    seqs = {seq_id: str(db[seq_id]["seq"]) for seq_id in db}

    workdir = path(workdir)
    if not workdir.exists():
        workdir.makedirs()
    infile = workdir / "in.fasta"

    with open(infile, "w") as f:
        for seq_id, seq in seqs.items():
            f.write(">{seq_id}\n".format(**locals()))
            f.write("{seq}\n".format(**locals()))

    if c > 7.0: n = "5"
    elif (c > 0.6) and (c <= 0.7): n = "4"
    elif (c > 0.5) and (c <= 0.6): n = "3"
    else: n = "2"

    outfile = path(workdir) / "out.cdhit"
    subprocess.check_call(["/net/noble/vol2/home/maxwl/Code/cdhit.git/trunk/cd-hit",
                           "-i", infile,
                           "-o", outfile,
                           "-c", str(c),
                           "-n", n,
                           "-M", "7000",
                           "-d", "0",
                           "-T", "1"])

    ret = []
    with open(outfile) as f:
        for line in f:
            if line[0] == ">":
                ret.append(int(line.strip()[1:]))

    return ret

# Like CD-HIT, but using my own implementation
# rather than calling the executable
def graph_cdhit_selection(db, sim, threshold=0.9, order_by_length=True):
    repset = set()
    if order_by_length:
        seq_ids_ordered = sorted(db.keys(), key=lambda seq_id: -len(str(db[seq_id]["seq"])))
    else:
        seq_ids_ordered = random.sample(db.keys(), len(db.keys()))
    next_log = 10
    for iteration_index, seq_id in enumerate(seq_ids_ordered):
        if iteration_index >= next_log:
            if not (logger is None):
                logger.info("graph_cdhit_selection {} / {}".format(iteration_index, len(seq_ids_ordered)))
            next_log *= 1.3
        covered = False
        for neighbor_seq_id, d in db[seq_id]["neighbors"].items():
            if (neighbor_seq_id in repset) and (sim_from_neighbor(sim, d) >= threshold):
                covered = True
                break
        if not covered:
            repset.add(seq_id)
    return sorted(list(repset))

# Use summaxacross to get a clustering on the sequences, then pick a random
# seq from each cluster
def cluster_selection(db, sim, k):
    assert(k < len(db.keys()))

    # Use k-medioids to get a clustering
    objective = summaxacross
    cluster_centers = []
    pq = [(-float("inf"), seq_id) for seq_id in db]
    objective_data = objective["base_data"](db, sim)
    cur_objective = 0
    while len(cluster_centers) < k:
        possible_diff, seq_id = heapq.heappop(pq)
        diff = objective["diff"](db, seq_id, sim, objective_data)
        next_diff = -pq[0][0]

        if diff >= next_diff:
            cluster_centers.append(seq_id)
            objective_data = objective["update"](db, seq_id, sim, objective_data)
            cur_objective += diff
        else:
            heapq.heappush(pq, (-diff, seq_id))

    clusters = objective_data["representatives"]

    # Choose a random sample from each cluster
    repset = []
    for i in range(k):
        clust = clusters.values()[i].keys()
        if len(clust) > 0:
            repset.append(random.choice(clust))
    return repset

# Use a clustering algorithm from sklearn
def sklearn_cluster_selection(db, db_seq_ids, db_seq_indices, sim, num_clusters_param, representative_type, cluster_type):
    #logger.info("Starting sklearn_cluster_selection: representative_type {representative_type} cluster_type {cluster_type} num_clusters_param {num_clusters_param}".format(**locals()))
    # Relevant clustering methods: Affinity prop, Spectral cluster, Agglomerative clustering
    logger.info("Starting creating similarity matrix...")
    logger.info("Memory usage: %s gb", float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000.0 )
    sims_matrix = numpy.zeros((len(db), len(db)))
    seq_ids = db.keys()
    for seq_id_index, seq_id in enumerate(seq_ids):
        for neighbor_seq_id, d in db[seq_id]["neighbors"].items():
            if not (neighbor_seq_id in db): continue
            neighbor_seq_id_index = db_seq_indices[neighbor_seq_id]
            s = sim_from_neighbor(sim, d)
            prev_s = sims_matrix[seq_id_index, neighbor_seq_id_index]
            if prev_s != 0:
                s = float(s + prev_s) / 2
            sims_matrix[seq_id_index, neighbor_seq_id_index] = s
            sims_matrix[neighbor_seq_id_index, seq_id_index] = s
    logger.info("Starting running clustering...")
    logger.info("Memory usage: %s gb", float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000.0 )
    linkage = "average" # FIXME
    if cluster_type == "agglomerative":
        num_clusters_param = int(num_clusters_param)
        model = sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters_param, affinity="precomputed", linkage=linkage)
    elif cluster_type == "affinityprop":
        model = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=num_clusters_param)
    elif cluster_type == "spectral":
        num_clusters_param = int(num_clusters_param)
        model = sklearn.cluster.SpectralClustering(n_clusters=num_clusters_param, affinity="precomputed")
    else:
        raise Exception("Unrecognized cluster_type: {cluster_type}".format(**locals()))
    try:
        cluster_ids = model.fit_predict(sims_matrix)
    except ValueError:
        # Spectral clustering breaks with ValueError when you ask for more clusters than rank of the matrix supports
        return random.sample(db.keys(), num_clusters_param)
    if numpy.isnan(cluster_ids[0]): return [] # AffinityProp sometimes breaks and just returns [nan]
    logger.info("Starting choosing repset and returning...")
    logger.info("Memory usage: %s gb", float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000.0 )
    cluster_ids_index = {}
    for i,c in enumerate(cluster_ids):
        if not (c in cluster_ids_index): cluster_ids_index[c] = []
        cluster_ids_index[c].append(i)
    repset = []
    if representative_type == "random":
        for c in cluster_ids_index:
            repset.append(db_seq_ids[random.choice(cluster_ids_index[c])])
    elif representative_type == "center":
        for c in cluster_ids_index:
            center_scores = {}
            cluster_seq_ids = set([db_seq_ids[seq_index] for seq_index in cluster_ids_index[c]])
            for seq_id in cluster_seq_ids:
                center_scores[seq_id] = sum([sim_from_neighbor(sim, d)
                                             for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items()
                                             if (neighbor_seq_id in cluster_seq_ids)])
            best_seq_id = max(center_scores.keys(), key=lambda seq_id: center_scores[seq_id])
            repset.append(best_seq_id)
    else:
        raise Exception("Unrecognized representative_type: {representative_type}".format(**locals()))
    return repset