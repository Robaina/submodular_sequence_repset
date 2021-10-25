#!/bin/env python

"""
1. Compatible with Python3
2. Reads percent identity from els-alipid output file instead of calling psi-blast
"""

import argparse
# import subprocess
# import math
import random
from pathlib import Path
import copy
# import math
import heapq
import logging

import pandas as pd
import numpy
from Bio import SeqIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--seqs", type=Path, required=True, help="Input sequences, fasta format")
    parser.add_argument("--pi", type=str, required=True, help="Input text file with PI from els-alipid")
    parser.add_argument("--mixture", type=float, default=0.5, help="Mixture parameter determining the relative weight of facility-location relative to sum-redundancy. Default=0.5")
    args = parser.parse_args()
    workdir = args.outdir

    assert args.mixture >= 0.0
    assert args.mixture <= 1.0

    if not workdir.exists():
        workdir.makedirs()

    # Logging
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s")
    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(workdir / "stdout.txt")
    fh.setLevel(logging.DEBUG) # >> this determines the file level
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)# >> this determines the output level
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
else:
    logger = logging.getLogger('log')

############################
# run_psiblast()
# ---------------
# Runs psiblast to get a similarity matrix between sequences.
# Input:
#   - workdir: directory for working files
#   - seqs: fasta file with sequences
# Output:
#   - db with sequence and similarity information.
# db: {seq_id: {
#               "neighbors": {neighbor_seq_id: {"log10_e": -9999.0, "pct_identical": 100.0}},
#               "in_neighbors": {neighbor_seq_id: {"log10_e": -9999.0, "pct_identical": 100.0}},
#               "scop": {level_id: [scop_ids]},
#               "seq": ""
#             }
#         }
# def run_psiblast(workdir, seqs):
    # Create psiblast db
    # if not (workdir / "db").exists():
    #     cmd = ["makeblastdb",
    #       "-in", seqs,
    #       "-input_type", "fasta",
    #       "-out", workdir / "db",
    #       "-dbtype", "prot"]
    #     logger.info(" ".join(cmd))
    #     subprocess.check_call(cmd)
    # # Run psiblast
    # if not (workdir / "psiblast_result.tab").exists():
    #     cmd = ["psiblast",
    #       "-query", seqs,
    #       "-db", workdir / "db",
    #       "-num_iterations", "6",
    #       "-outfmt", "6 qseqid sseqid pident length mismatch evalue bitscore",
    #       "-seg", "yes",
    #       "-out", workdir / "psiblast_result.tab"
    #     ]
    #     logger.info(" ".join(cmd))
    #     subprocess.check_call(cmd)
    # Read psiblast output
    # db = {}
    # fasta_sequences = SeqIO.parse(open(seqs),'fasta')
    # for seq in fasta_sequences:
    #     seq_id = seq.id
    #     db[seq_id] = {"neighbors": {}, "in_neighbors": {}, "seq": seq.seq}
    # with open(workdir / "psiblast_result.tab", "r") as f:
    #     for line in f:
    #         if line.strip() == "": continue
    #         if line.startswith("Search has CONVERGED!"): continue
    #         line = line.split()
    #         seq_id1 = line[0]
    #         seq_id2 = line[1]
    #         pident = float(line[2])
    #         evalue = line[5]
    #         evalue = float(evalue) + 1e-80 # Here, we would have number of identical aligned pairs in esl-alipd instead of evalue
    #         log10_e = math.log10(float(evalue))
    #         if float(evalue) <= 1e-2:
    #             db[seq_id2]["neighbors"][seq_id1] = {"log10_e": log10_e, "pct_identical": pident}
    #             db[seq_id1]["in_neighbors"][seq_id2] = {"log10_e": log10_e, "pct_identical": pident}
    # return db

def get_pident(pi_file, seqs):
    """
    Parse esl-alipid output file
    """
    df = pd.read_csv(
        pi_file, sep='\s+',
        skiprows=1, header=None
    )
    # df = df.applymap(lambda cell: cell.re)
    df.columns = [
        'seqname1', 'seqname2', '%id', 'nid', 'denomid', '%match', 'nmatch', 'denommatch'
        ]
    
    db = {}
    fasta_sequences = SeqIO.parse(seqs,'fasta') # open not needed (at least in python3)
    # Initialize dict
    for seq in fasta_sequences:
        seq_id = seq.id
        # print(seq_id)
        db[seq_id] = {"neighbors": {}, "in_neighbors": {}, "seq": seq.seq}
    
    for i, row in df.iterrows():
        seq_id1 = row.seqname1.split('/')[0]
        seq_id2 = row.seqname2.split('/')[0]
        # print(seq_id1, seq_id2)
        log10_e = -100
        pident = row['%id']
        # Originally filtering pairs with evalue > 1e-2 (perhaps nid, %match)
        if seq_id1 in db.keys() and seq_id2 in db.keys():
            db[seq_id2]["neighbors"][seq_id1] = {"log10_e": log10_e, "pct_identical": pident}
            db[seq_id1]["in_neighbors"][seq_id2] = {"log10_e": log10_e, "pct_identical": pident}

    return db


###############################################################
###############################################################
# Submodular optimization functions
###############################################################
###############################################################

#############################
# Similarity functions
#############################

def sim_from_db(db, sim, seq_id1, seq_id2):
    d = db[seq_id1]["neighbors"][seq_id2]
    return sim_from_neighbor(sim, d)

def sim_from_neighbor(sim, d):
    return sim(d["log10_e"], d["pct_identical"])

def fraciden(log10_e, pct_identical):  # Not using log10_e at all
    return float(pct_identical) / 100

def rankpropsim(log10_e, pct_identical):
    return numpy.exp(-numpy.power(10, log10_e) / 100.0)

def rankpropsim_loge(log10_e, pct_identical):
    return numpy.exp(-log10_e / 100.0)

def logloge(log10_e, pct_identical):
    if (-log10_e) <= 0.1:
        return 0.0
    elif (-log10_e) >= 1000:
        return 3.0
    else:
        return numpy.log10(-log10_e)

def oneprankpropsim(log10_e, pct_identical):
    return 1.0 + 1e-3 * rankpropsim_loge(log10_e, pct_identical)

def prodevaliden(log10_e, pct_identical):
    return fraciden(log10_e, pct_identical) * logloge(log10_e, pct_identical) / 3

def one(log10_e, pct_identical):
    return 1.0

def p90(log10_e, pct_identical):
    return float(pct_identical >= 0.9)


#############################
# Objective functions
# -----------------
# An objective is a dictionary
# {"eval": db, seq_ids, sim -> float, # The value of an objective function
#  "diff": db, seq_ids, sim, data -> float, # The difference in value after adding seq_id
#  "negdiff": db, seq_ids, sim, data -> float, # The difference in value after removing seq_id
#  "update": db, seq_ids, sim, data -> data, # Update the data structure to add seq_id as a representative
#  "negupdate": db, seq_ids, sim, data -> data, # Update the data structure to remove seq_id as a representative
#  "base_data": db, sim -> data, # The data structure corresponding to no representatives chosen
#  "full_data": db, sim -> data, # The data structure corresponding to all representatives chosen
#  "name": name}
# db: Database
# sim: Similiarity function
# data: function-specific data structure which may be modified over the course of an optimization algorithm
#############################

######################
# summaxacross
# AKA facility location
######################

def summaxacross_eval(db, seq_ids, sim):
    max_sim = {seq_id:0 for seq_id in db}
    for chosen_seq_id in seq_ids:
        for neighbor_seq_id, d in db[chosen_seq_id]["in_neighbors"].items():
            if neighbor_seq_id in max_sim:
                sim_val = sim(d["log10_e"], d["pct_identical"])
                if sim_val > max_sim[neighbor_seq_id]:
                    max_sim[neighbor_seq_id] = sim_val
                #max_sim[neighbor_seq_id] = max(max_sim[neighbor_seq_id], sim(d["log10_e"], d["pct_identical"]))
            else:
                pass
                #raise Exception("Found node with neighbor not in set")

    return sum(max_sim.values())

# summaxacross data:
# Who each example is represented by
# "examples": {seq_id: (representive, val) }
# Who each represntative represents
# "representatives" {seq_id: {example: val}}

summaxacross_base_data = lambda db, sim: {"examples": {seq_id: (None, 0) for seq_id in db},
                                              "representatives": {}}

summaxacross_full_data = lambda db, sim: {"examples": {seq_id: (seq_id, sim_from_db(db, sim, seq_id, seq_id)) for seq_id in db},
                                              "representatives": {seq_id: {seq_id: sim_from_db(db, sim, seq_id, seq_id)} for seq_id in db}}

def summaxacross_diff(db, seq_id, sim, data):
    diff = 0
    for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
        if neighbor_seq_id in data["examples"]:
            sim_val = sim(d["log10_e"], d["pct_identical"])
            if sim_val > data["examples"][neighbor_seq_id][1]:
                diff += sim_val - data["examples"][neighbor_seq_id][1]
        else:
            pass
            #raise Exception("Found node with neighbor not in set")
    return diff

def summaxacross_update(db, seq_id, sim, data):
    data = copy.deepcopy(data)
    data["representatives"][seq_id] = {}
    for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
        if neighbor_seq_id in data["examples"]:
            sim_val = sim(d["log10_e"], d["pct_identical"])
            if sim_val > data["examples"][neighbor_seq_id][1]:
                data["examples"][neighbor_seq_id] = (seq_id, sim_val)
                data["representatives"][seq_id][neighbor_seq_id] = sim_val
        else:
            pass
            #raise Exception("Found node with neighbor not in set")
    return data

# O(D^2)
def summaxacross_negdiff(db, seq_id, sim, data):
    diff = 0
    # For each neighbor_seq_id that was previously represented by seq_id
    new_representatives = (set(data["representatives"].keys()) - set([seq_id]))
    for neighbor_seq_id, d in data["representatives"][seq_id].items():
        # Find the next-best representative for neighbor_seq_id
        candidate_ids = set(db[neighbor_seq_id]["neighbors"].keys()) & new_representatives
        if len(candidate_ids) == 0:
            diff += -d
        else:
            best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, neighbor_seq_id, x))
            diff += sim_from_db(db, sim, neighbor_seq_id, best_id) - d
    return diff

# O(D^2)
def summaxacross_negupdate(db, seq_id, sim, data):
    data = copy.deepcopy(data)
    new_representatives = (set(data["representatives"].keys()) - set([seq_id]))
    for neighbor_seq_id, d in data["representatives"][seq_id].items():
        # Find the next-best representative for neighbor_seq_id
        candidate_ids = set(db[neighbor_seq_id]["neighbors"].keys()) & new_representatives
        if len(candidate_ids) == 0:
            data["examples"][neighbor_seq_id] = (None, 0)
        else:
            best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, neighbor_seq_id, x))
            data["examples"][neighbor_seq_id] = (best_id, sim_from_db(db, sim, neighbor_seq_id, best_id))
            data["representatives"][best_id][neighbor_seq_id] = sim_from_db(db, sim, neighbor_seq_id, best_id)
    del data["representatives"][seq_id]
    return data

summaxacross = {"eval": summaxacross_eval,
          "diff": summaxacross_diff,
          "negdiff": summaxacross_negdiff,
          "update": summaxacross_update,
          "negupdate": summaxacross_negupdate,
          "base_data": summaxacross_base_data,
          "full_data": summaxacross_full_data,
          "name": "summaxacross"}

######################
# summaxwithin
# AKA negfacloc
######################

def summaxwithin_eval(db, seq_ids, sim):
    max_sim = {seq_id:0 for seq_id in seq_ids}
    for chosen_seq_id in seq_ids:
        for neighbor_seq_id, d in db[chosen_seq_id]["in_neighbors"].items():
            if neighbor_seq_id == chosen_seq_id: continue
            if neighbor_seq_id in max_sim:
                sim_val = sim(d["log10_e"], d["pct_identical"])
                if sim_val > max_sim[neighbor_seq_id]:
                    max_sim[neighbor_seq_id] = sim_val
            else:
                pass
    return -sum(max_sim.values())

# summaxwithin data:
# Who each example is represented by
# "examples": {seq_id: (representive, val) }
# Who each represntative represents
# "representatives" {seq_id: {example: val}}

summaxwithin_base_data = lambda db, sim: {"examples": {seq_id: (None, 0) for seq_id in db},
                                          "representatives": {}}

def summaxwithin_full_data(db, sim):
    data = {}
    data["examples"] = {}
    data["representatives"] = {seq_id: {} for seq_id in db}
    for seq_id in db:
        neighbors = {neighbor_seq_id: d for neighbor_seq_id,d in db[seq_id]["neighbors"].items() if neighbor_seq_id != seq_id}
        if len(neighbors) == 0:
            data["examples"][seq_id] = (None, 0)
        else:
            d = max(neighbors.items(), key=lambda d: sim_from_neighbor(sim, d[1]))
            data["examples"][seq_id] = (d[0], sim_from_neighbor(sim, d[1]))
            data["representatives"][d[0]][seq_id] = sim_from_neighbor(sim, d[1])
    return data

def summaxwithin_diff(db, seq_id, sim, data):
    diff = 0
    # Difference introduced in other representatives
    for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
        if neighbor_seq_id == seq_id: continue
        if neighbor_seq_id in data["representatives"]:
            sim_val = sim(d["log10_e"], d["pct_identical"])
            if sim_val > data["examples"][neighbor_seq_id][1]:
                # adding a penalty of sim_val, removing old penalty
                diff -= sim_val - data["examples"][neighbor_seq_id][1]
    # Difference from adding this representative
    neighbors = {neighbor_seq_id: d for neighbor_seq_id,d in db[seq_id]["neighbors"].items() if neighbor_seq_id != seq_id}
    if len(neighbors) == 0:
        diff -= 0
    else:
        d = max(neighbors.items(), key=lambda d: sim_from_neighbor(sim, d[1]))
        diff -= sim_from_neighbor(sim, d[1])
    return diff

def summaxwithin_update(db, seq_id, sim, data):
    data = copy.deepcopy(data)
    # Find best repr for seq_id
    candidate_ids = (set(db[seq_id]["neighbors"].keys()) & set(data["representatives"].keys())) - set([seq_id])
    if len(candidate_ids) == 0:
        data["examples"][seq_id] = (None, 0)
    else:
        best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, seq_id, x))
        data["examples"][seq_id] = (best_id, sim_from_db(db, sim, seq_id, best_id))
        data["representatives"][best_id][seq_id] = sim_from_db(db, sim, seq_id, best_id)
    # Find ids represented by seq_id
    data["representatives"][seq_id] = {}
    for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
        if neighbor_seq_id in data["examples"]:
            if neighbor_seq_id == seq_id: continue
            sim_val = sim(d["log10_e"], d["pct_identical"])
            if sim_val > data["examples"][neighbor_seq_id][1]:
                data["examples"][neighbor_seq_id] = (seq_id, sim_val)
                data["representatives"][seq_id][neighbor_seq_id] = sim_val
        else:
            pass
            #raise Exception("Found node with neighbor not in set")
    return data

# O(D^2)
def summaxwithin_negdiff(db, seq_id, sim, data):
    diff = 0
    # Difference introduced in other representatives
    # For each neighbor_seq_id that was previously represented by seq_id
    new_representatives = (set(data["representatives"].keys()) - set([seq_id]))
    for neighbor_seq_id, d in data["representatives"][seq_id].items():
        # Find the next-best representative for neighbor_seq_id
        candidate_ids = set(db[neighbor_seq_id]["neighbors"].keys()) & new_representatives
        if len(candidate_ids) == 0:
            diff += d # removing a penalty of -d
        else:
            best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, neighbor_seq_id, x))
            # removing a penalty of d, adding a new penalty of -sim(neighbor, best)
            diff += d - sim_from_db(db, sim, neighbor_seq_id, best_id)
    # Difference from adding this representative
    diff += data["examples"][seq_id][1] # removing a penalty of -sim
    return diff

# O(D^2)
def summaxwithin_negupdate(db, seq_id, sim, data):
    data = copy.deepcopy(data)
    del data["examples"][seq_id]
    new_representatives = (set(data["representatives"].keys()) - set([seq_id]))
    for neighbor_seq_id, d in data["representatives"][seq_id].items():
        # Find the next-best representative for neighbor_seq_id
        candidate_ids = set(db[neighbor_seq_id]["neighbors"].keys()) & new_representatives
        if len(candidate_ids) == 0:
            data["examples"][neighbor_seq_id] = (None, 0)
        else:
            best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, neighbor_seq_id, x))
            data["examples"][neighbor_seq_id] = (best_id, sim_from_db(db, sim, neighbor_seq_id, best_id))
            data["representatives"][best_id][neighbor_seq_id] = sim_from_db(db, sim, neighbor_seq_id, best_id)
    del data["representatives"][seq_id]
    return data

summaxwithin = {"eval": summaxwithin_eval,
                "diff": summaxwithin_diff,
                "negdiff": summaxwithin_negdiff,
                "update": summaxwithin_update,
                "negupdate": summaxwithin_negupdate,
                "base_data": summaxwithin_base_data,
                "full_data": summaxwithin_full_data,
                "name": "summaxwithin"}


######################
# sumsumwithin
######################

def bisim(db, sim, seq_id1, seq_id2):
    ret = 0
    if seq_id2 in db[seq_id1]["neighbors"]:
        d = db[seq_id1]["neighbors"][seq_id2]
        ret += sim(d["log10_e"], d["pct_identical"])
    if seq_id1 in db[seq_id2]["neighbors"]:
        d = db[seq_id2]["neighbors"][seq_id1]
        ret += sim(d["log10_e"], d["pct_identical"])
    return ret

def sumsumwithin_eval(db, seq_ids, sim):
    seq_ids = set(seq_ids)
    s = 0
    for chosen_id in seq_ids:
        for neighbor, d in db[chosen_id]["neighbors"].items():
            if chosen_id == neighbor: continue
            if neighbor in seq_ids:
                s += -sim(d["log10_e"], d["pct_identical"])
    return s

sumsumwithin_base_data = lambda db, sim: set()
sumsumwithin_full_data = lambda db, sim: set(db.keys())

def sumsumwithin_diff(db, seq_id, sim, data):
    diff = 0
    data = data | set([seq_id])
    for neighbor, d in db[seq_id]["neighbors"].items():
        if seq_id == neighbor: continue
        if not (neighbor in data): continue
        diff += -sim_from_neighbor(sim, d)
        #neighbor_bisim = bisim(db, sim, seq_id, neighbor)
        #diff += -neighbor_bisim
    for neighbor, d in db[seq_id]["in_neighbors"].items():
        if seq_id == neighbor: continue
        if not (neighbor in data): continue
        diff += -sim_from_neighbor(sim, d)
    return diff

def sumsumwithin_update(db, seq_id, sim, data):
    data.add(seq_id)
    return data

def sumsumwithin_negdiff(db, seq_id, sim, data):
    diff = 0
    #data = data - set([seq_id])
    for neighbor, d in db[seq_id]["neighbors"].items():
        if seq_id == neighbor: continue
        if not (neighbor in data): continue
        #neighbor_bisim = bisim(db, sim, seq_id, neighbor)
        #diff -= -neighbor_bisim
        diff += sim_from_neighbor(sim, d) # removing a penalty
    for neighbor, d in db[seq_id]["in_neighbors"].items():
        if seq_id == neighbor: continue
        if not (neighbor in data): continue
        diff += sim_from_neighbor(sim, d) # removing a penalty
    return diff

def sumsumwithin_negupdate(db, seq_id, sim, data):
    data.remove(seq_id)
    return data

sumsumwithin = {"eval": sumsumwithin_eval,
          "diff": sumsumwithin_diff,
          "negdiff": sumsumwithin_negdiff,
          "update": sumsumwithin_update,
          "negupdate": sumsumwithin_negupdate,
          "base_data": sumsumwithin_base_data,
          "full_data": sumsumwithin_full_data,
          "name": "sumsumwithin"}

######################
# sumsumacross
######################

def sumsumacross_eval(db, seq_ids, sim):
    seq_ids = set(seq_ids)
    s = 0
    for chosen_id in seq_ids:
        for neighbor, d in db[chosen_id]["neighbors"].items():
            s += -sim(d["log10_e"], d["pct_identical"])
    return s

sumsumacross_base_data = lambda db, sim: None
sumsumacross_full_data = lambda db, sim: None

def sumsumacross_diff(db, seq_id, sim, data):
    diff = 0
    for neighbor, d in db[seq_id]["neighbors"].items():
        if seq_id == neighbor: continue
        diff += -sim(d["log10_e"], d["pct_identical"])
    return diff

def sumsumacross_negdiff(db, seq_id, sim, data):
    diff = 0
    for neighbor, d in db[seq_id]["neighbors"].items():
        if seq_id == neighbor: continue
        diff -= -sim(d["log10_e"], d["pct_identical"])
    return diff

def sumsumacross_update(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

def sumsumacross_negupdate(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

sumsumacross = {"eval": sumsumacross_eval,
          "diff": sumsumacross_diff,
          "negdiff": sumsumacross_negdiff,
          "update": sumsumacross_update,
          "negupdate": sumsumacross_negupdate,
          "base_data": sumsumacross_base_data,
          "full_data": sumsumacross_full_data,
          "name": "sumsumacross"}

######################
# MixtureObjective
# ------------------------
# Create a mixture objective with:
# MixtureObjective([summaxacross, sumsumwithin], [0.1, 1.2])
# Must be used with a sim of the form
# [sim1, sim2]
# (same number of sims as objectives)
######################

class MixtureObjective(object):
    def __init__(self, objectives, weights):
        self.objectives = objectives
        self.weights = weights
        self.name = "mix-" + "-".join(["{0}({1})".format(objective["name"], self.weights[i]) for i,objective in enumerate(self.objectives)])

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __contains__(self, item):
        all_contain = True
        for i, objective in enumerate(self.objectives):
            all_contain = all_contain and (item in objective)
        return all_contain

    def eval(self, db, seq_ids, sims):
        s = 0
        for i, objective in enumerate(self.objectives):
            s += self.weights[i]*objective["eval"](db, seq_ids, sims[i])
        return s

    def diff(self, db, seq_id, sims, datas):
        s = 0
        for i, objective in enumerate(self.objectives):
            s += self.weights[i]*objective["diff"](db, seq_id, sims[i], datas[i])
        return s

    def negdiff(self, db, seq_id, sims, datas):
        s = 0
        for i, objective in enumerate(self.objectives):
            s += self.weights[i]*objective["negdiff"](db, seq_id, sims[i], datas[i])
        return s

    def update(self, db, seq_id, sims, datas):
        new_datas = []
        for i, objective in enumerate(self.objectives):
            new_datas.append(objective["update"](db, seq_id, sims[i], datas[i]))
        return new_datas

    def negupdate(self, db, seq_id, sims, datas):
        new_datas = []
        for i, objective in enumerate(self.objectives):
            new_datas.append(objective["negupdate"](db, seq_id, sims[i], datas[i]))
        return new_datas

    def base_data(self, db, sims):
        datas = []
        for i, objective in enumerate(self.objectives):
            datas.append(objective["base_data"](db, sims[i]))
        return datas

    def full_data(self, db, sims):
        datas = []
        for i, objective in enumerate(self.objectives):
            datas.append(objective["full_data"](db, sims[i]))
        return datas


#############################
# Optimization algorithms
# -------------------------
# Each returns either a specific
# subset or an order.
#############################


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


###############################################################
###############################################################
# Utility functions
###############################################################
###############################################################


# Either randomly adds or removes elements from a repset
# to match the target size
# def random_topup(db, rs, target_repset_size):
#     target_repset_size = int(target_repset_size)
#     if len(rs) > target_repset_size:
#         return random.sample(rs, target_repset_size)
#     elif len(rs) < target_repset_size:
#         return list(rs) + list(random.sample(set(db.keys()) - set(rs), target_repset_size-len(rs)))
#     else:
#         return rs

#########################################
# binary_search_repset_size
# ----------------------------
# Performs binary search to find a repset of size
# exactly equal to target.
# Usage:
# db = input database
# f = A function such that f(param) -> repset
# target = target repset size
# param_lower = lowest value of param to check. We assume that higher param <=> larger repset
# param_upper = highest value of param to check
# tolerance = If we find a repset of target<=size<target+tolerance, return sample(repset, target)
# max_iters = maximum binary search iterations to attempt
#########################################

# def binary_search_repset_size(db, f, target, param_lower, param_upper, max_iters=10, tolerance=3, final_tolerance=float("inf")):
#     # find range that includes target
#     # use iters in increments of 2 because we have to compute f() twice at each iter
#     for iter_index in range(0,int(float(max_iters)/2),2):
#         size_upper = len(f(param_upper))
#         size_lower = len(f(param_lower))
#         if (size_lower < target and size_upper < target) or (size_lower > target and size_upper > target):
#             param_upper += param_upper - param_lower
#             param_lower -= param_upper - param_lower
#         else:
#             break
#     range_search_iters = iter_index
#     # find target within range
#     for iter_index in range(range_search_iters, max_iters):
#         try_param = float(param_upper + param_lower) / 2
#         rs = f(try_param)
#         print param_lower, param_upper, try_param, len(rs), target
#         if abs(len(rs) - target) <= tolerance:
#             ret_rs = rs
#             break
#         elif len(rs) > target:
#             param_upper = try_param
#         elif len(rs) < target:
#             param_lower = try_param
#     if abs(len(rs) - final_tolerance) <= final_tolerance:
#         ret_rs = rs
#     else:
#         len_rs = len(rs)
#         raise(Exception("binary_search_repset_size failed to find a suitable param. Final parameters: param_lower={param_lower}; param_upper={param_upper}; len(rs)={len_rs}; target={target}".format(**locals())))
#     ret_rs = random_topup(db, ret_rs, target)
#     return {"rs": ret_rs, "iters": iter_index+1, "param": try_param}

# def binary_search_order(low_x, low_y, high_x, high_y):
#     # pq entry: (-abs(low_y-high_y), low_x, low_y, high_x, low_y)
#     pq = []

#     def binary_search_order_next(mid_x, mid_y, low_x, low_y, high_x, high_y):

#         # Update heap from last time
#         heapq.heappush(pq, (-abs(mid_y-low_y), low_x, low_y, mid_x, mid_y))
#         heapq.heappush(pq, (-abs(high_y-mid_y), mid_x, mid_y, high_x, high_y))

#         # Return next x to try
#         negdiff, low_x, low_y, high_x, high_y = heapq.heappop(pq)
#         mid_x = float(high_x + low_x) / 2
#         return mid_x, low_x, low_y, high_x, high_y

#     return binary_search_order_next

# def binary_parameter_search(f, low_x, high_x, num_iterations=30):
#     low_y = f(low_x)
#     if low_y is None:
#         raise Exception("low_y is None -- maybe faulty return value from f()")
#     high_y = f(high_x)
#     get_next = binary_search_order(low_x, low_y, high_x, high_y)
#     next_x = (high_x + low_x) / 2
#     for i in range(num_iterations):
#         next_x, low_x, low_y, high_x, high_y = get_next(next_x, f(next_x), low_x, low_y, high_x, high_y)


###############################################################
###############################################################
# Run optimization and output
###############################################################
###############################################################

if __name__ == "__main__":
    # db = run_psiblast(workdir, args.seqs)  
    db = get_pident(pi_file=args.pi, seqs=args.seqs) # Make db with PI from esl-alipid
    objective = MixtureObjective([summaxacross, sumsumwithin], [args.mixture, 1.0-args.mixture])
    logger.info("-----------------------")
    logger.info("Starting mixture of summaxacross and sumsumwithin with weight %s...", args.mixture)
    sim, sim_name = ([fraciden, fraciden], "fraciden-fraciden")
    repset_order = accelerated_greedy_selection(db, objective, sim) # Call main algorithm

    with open(workdir / "repset.txt", "w") as f:
        for seq_id in repset_order:
            f.write(seq_id)
            f.write("\n")

