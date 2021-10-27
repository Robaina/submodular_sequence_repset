#!/bin/env python

"""
1. Compatible with Python3
2. Reads percent identity from els-alipid output file instead of calling psi-blast
"""

import argparse
from pathlib import Path
import logging

from repset.database import get_pident_from_file
from repset.objectives import MixtureObjective, summaxacross, sumsumwithin
from repset.similarity import fraciden
from repset.optimization import accelerated_greedy_selection



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find representative sets of sequences of given size")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--seqs", type=Path, required=True, help="Input sequences, fasta format")
    parser.add_argument("--pi", type=str, required=True, help="Input text file with PI from els-alipid")
    parser.add_argument("--mixture", type=float, default=0.5, help=("Mixture parameter determining the relative "
                                                                    "weight of facility-location relative to sum-redundancy. "
                                                                    "Default=0.5"))
    parser.add_argument("--size", type=int, default=float("inf"), help="Repset size. Default=inf")
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


if __name__ == "__main__":
    
    print('Reading PI database...')
    db = get_pident_from_file(pi_file=args.pi) #, seqs=args.seqs) # Make db with PI from esl-alipid
    print('Finished building database...')
    objective = MixtureObjective([summaxacross, sumsumwithin], [args.mixture, 1.0-args.mixture])
    logger.info("-----------------------")
    logger.info("Starting mixture of summaxacross and sumsumwithin with weight %s...", args.mixture)
    sim, sim_name = ([fraciden, fraciden], "fraciden-fraciden")
    repset_order = accelerated_greedy_selection(db, objective, sim, repset_size=args.size) # Call main algorithm

    with open(workdir / "repset.txt", "w") as f:
        for seq_id in repset_order:
            f.write(seq_id)
            f.write("\n")