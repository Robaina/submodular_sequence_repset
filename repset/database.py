#!/bin/env python

"""
Create pairwise percent identity database
"""

import subprocess
import math

import pandas as pd
from Bio import SeqIO


def run_psiblast(workdir, seqs, logger):
    # Create psiblast db
    if not (workdir / "db").exists():
        cmd = ["makeblastdb",
          "-in", str(seqs),
          "-input_type", "fasta",
          "-out", str(workdir / "db"),
          "-dbtype", "prot"]
        logger.info(" ".join(cmd))
        subprocess.check_call(cmd)
    # Run psiblast
    if not (workdir / "psiblast_result.tab").exists():
        cmd = ["psiblast",
          "-query", str(seqs),
          "-db", str(workdir / "db"),
          "-num_iterations", "6",
          "-outfmt", "6 qseqid sseqid pident length mismatch evalue bitscore",
          "-seg", "yes",
          "-out", str(workdir / "psiblast_result.tab")
        ]
        logger.info(" ".join(cmd))
        subprocess.check_call(cmd)
    # Read psiblast output
    db = {}
    fasta_sequences = SeqIO.parse(open(seqs),'fasta')
    for seq in fasta_sequences:
        seq_id = seq.id
        db[seq_id] = {"neighbors": {}, "in_neighbors": {}, "seq": seq.seq}
    with open(workdir / "psiblast_result.tab", "r") as f:
        for line in f:
            if line.strip() == "": continue
            if line.startswith("Search has CONVERGED!"): continue
            line = line.split()
            seq_id1 = line[0]
            seq_id2 = line[1]
            pident = float(line[2])
            evalue = line[5]
            evalue = float(evalue) + 1e-80 # Here, we would have number of identical aligned pairs in esl-alipd instead of evalue
            log10_e = math.log10(float(evalue))
            if float(evalue) <= 1e-2:
                db[seq_id2]["neighbors"][seq_id1] = {"log10_e": log10_e, "pct_identical": pident}
                db[seq_id1]["in_neighbors"][seq_id2] = {"log10_e": log10_e, "pct_identical": pident}
    return db

def get_pident_from_file(pi_file):
    """
    Parse esl-alipid output file
    """
    print('Building dataframe')
    df = pd.read_csv(
        pi_file, sep='\s+',
        skiprows=1, header=None
    )
    df.columns = [
        'seqname1', 'seqname2', '%id', 'nid', 'denomid', '%match', 'nmatch', 'denommatch'
        ]
    print('Dataframe built')
    
    db = {}
    for i, row in df.iterrows():
        seq_id1 = row.seqname1.split('/')[0]
        seq_id2 = row.seqname2.split('/')[0]
        log10_e = -100
        pident = row['%id']

        # Originally filtering pairs with evalue > 1e-2 (perhaps nid, %match)
        db[seq_id1] = {"neighbors": {}, "in_neighbors": {}}
        db[seq_id2] = {"neighbors": {}, "in_neighbors": {}}

        # if seq_id1 in db.keys() and seq_id2 in db.keys():
        db[seq_id2]["neighbors"][seq_id1] = {"log10_e": log10_e, "pct_identical": pident}
        db[seq_id1]["in_neighbors"][seq_id2] = {"log10_e": log10_e, "pct_identical": pident}

    del df
    return db