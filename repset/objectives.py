"""
Objective functions
-----------------
An objective is a dictionary
{
    "eval": db, seq_ids, sim -> float, # The value of an objective function
    "diff": db, seq_ids, sim, data -> float, # The difference in value after adding seq_id
    "negdiff": db, seq_ids, sim, data -> float, # The difference in value after removing seq_id
    "update": db, seq_ids, sim, data -> data, # Update the data structure to add seq_id as a representative
    "negupdate": db, seq_ids, sim, data -> data, # Update the data structure to remove seq_id as a representative
    "base_data": db, sim -> data, # The data structure corresponding to no representatives chosen
    "full_data": db, sim -> data, # The data structure corresponding to all representatives chosen
    "name": name
}

db: Database
sim: Similiarity function
data: function-specific data structure which may be modified over the course of an optimization algorithm
"""

import copy
from repset.similarity import *


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
# minmaxacross
# Most comparable to CD-HIT
# Eval only
######################

def minmaxacross_eval(db, seq_ids, sim):
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

    return min(max_sim.values())

minmaxacross = {"eval": minmaxacross_eval,
          "name": "minmaxacross"}

######################
# maxmaxwithin
# Also comparable to CD-HIT
# Eval only
######################

def maxmaxwithin_eval(db, seq_ids, sim):
    max_sim = float("-inf")
    seq_ids_set = set(seq_ids)
    for chosen_seq_id in seq_ids:
        for neighbor_seq_id, d in db[chosen_seq_id]["neighbors"].items():
            if (neighbor_seq_id in seq_ids_set) and (neighbor_seq_id != chosen_seq_id):
                sim_val = sim(d["log10_e"], d["pct_identical"])
                if sim_val > max_sim:
                    max_sim = sim_val
    return -max_sim

maxmaxwithin = {"eval": maxmaxwithin_eval,
                "name": "maxmaxwithin"}

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
# lengthobj
######################

def lengthobj_eval(db, seq_ids, sim):
    s = 0
    for chosen_id in seq_ids:
        s += len(db[chosen_id]["seq"])
    return s

lengthobj_base_data = lambda db, sim: None
lengthobj_full_data = lambda db, sim: None

def lengthobj_diff(db, seq_id, sim, data):
    return len(db[seq_id]["seq"])

def lengthobj_negdiff(db, seq_id, sim, data):
    return len(db[seq_id]["seq"])

def lengthobj_update(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

def lengthobj_negupdate(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

lengthobj = {"eval": lengthobj_eval,
          "diff": lengthobj_diff,
          "negdiff": lengthobj_negdiff,
          "update": lengthobj_update,
          "negupdate": lengthobj_negupdate,
          "base_data": lengthobj_base_data,
          "full_data": lengthobj_full_data,
          "name": "lengthobj"}



######################
# graphcut
######################

graphcut = {"eval": lambda *args: sumsumacross_eval(*args) + sumsumwithin_eval(*args),
          "diff": lambda *args: sumsumacross_diff(*args) + sumsumwithin_diff(*args),
          "negdiff": lambda *args: sumsumacross_negdiff(*args) + sumsumwithin_negdiff(*args),
          "update": sumsumwithin_update,
          "negupdate": sumsumwithin_negupdate,
          "base_data": sumsumwithin_base_data,
          "full_data": sumsumwithin_full_data,
          "name": "graphcut"}

######################
# uniform
######################
def uniform_eval(db, seq_ids, sim):
    return len(seq_ids)

uniform_base_data = lambda db, sim: None
uniform_full_data = lambda db, sim: None

def uniform_diff(db, seq_id, sim, data):
    return 1

def uniform_negdiff(db, seq_id, sim, data):
    return -1

def uniform_update(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

def uniform_negupdate(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

uniform = {"eval": uniform_eval,
          "diff": uniform_diff,
          "negdiff": uniform_negdiff,
          "update": uniform_update,
          "negupdate": uniform_negupdate,
          "base_data": uniform_base_data,
          "full_data": uniform_full_data,
          "name": "uniform"}


####################
# Objective classes
####################

class SetCover(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.name = "setcover-" + str(self.threshold)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def eval(self, db, seq_ids, sim):
        s = 0
        for seq_id in db:
            for neighbor_seq_id, d in db[seq_id]["neighbors"].items():
                if sim_from_neighbor(sim, d) >= self.threshold:
                    s += 1
                    break
        return s

    def diff(self, db, seq_id, sim, data):
        s = 0
        for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
            if neighbor_seq_id in data:
                if not data[neighbor_seq_id]:
                    if sim_from_neighbor(sim, d) >= self.threshold:
                        s += 1
        return s

    def negdiff(self, db, seq_id, sim, data):
        raise Exception()

    def update(self, db, seq_id, sim, data):
        for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
            if neighbor_seq_id in data:
                if not data[neighbor_seq_id]:
                    if sim_from_neighbor(sim, d) >= self.threshold:
                        data[neighbor_seq_id] = True
        return data

    def negupdate(self, db, seq_id, sim, datas):
        raise Exception()

    def base_data(self, db, sim):
        return {seq_id: False for seq_id in db}

    def full_data(self, db, sim):
        return {seq_id: (sim_from_db(db, sim, seq_id, seq_id) >= self.threshold)
                for seq_id in db}


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