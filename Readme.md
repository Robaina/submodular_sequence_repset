
# SAGA annotation automated label interpretation #


This script selects a representative set of protein or DNA sequences from a larger set using submodular optimization. See [this manuscript](https://doi.org/10.1101/051201) for more information.

Required Python packages:

* BioPython (FIXME)
* scikit-learn
* path (https://pypi.python.org/pypi/forked-path)

usage: repset.py [-h] --outdir OUTDIR --seqs SEQS [--mixture MIXTURE]

optional arguments:
  -h, --help         show this help message and exit
  --outdir OUTDIR    Output directory
  --seqs SEQS        Input sequences, fasta format
  --mixture MIXTURE  Mixture parameter determining the relative weight of
                     facility-location relative to sum-redundancy. Default=0.5





