"""
Similarity functions
"""

import numpy


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