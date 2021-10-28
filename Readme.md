# UPDATE (October, 2021)

In this fork, I have restructured the project. Major changes include:

1) Ported to Python3
2) New function to get pairwise sequence percent identities from the output file of ```esl-alipid```, which computes identities from a MSA file (MSA file may be obtained with an aligner such as muscle or mafft). The old function ```run_psiblast()``` is maintained in the project. I made this change to speed up the algorithm, since psiblast takes long time to compute percent identities when querying a large list of sequences.
3) Code and project refactoring. Split the original script into modules.

Dependencies may be installed in a new conda environment like this:
```
conda env create -f environment.yml
```

## Usage
```
usage: repset.py [-h] --outdir OUTDIR --seqs SEQS [--pi PI] [--mixture MIXTURE] [--size SIZE]

Find representative sets of sequences of given size

optional arguments:
  -h, --help         show this help message and exit
  --outdir OUTDIR    Output directory
  --seqs SEQS        Input sequences, fasta format
  --pi PI            Input text file with PI from els-alipid. If not provided, percent identities are computed per psiblast.
  --mixture MIXTURE  Mixture parameter determining the relative weight of facility-location relative to sum-redundancy. Default=0.5
  --size SIZE        Repset size. Default=inf
```

# Protein sequence representative set selection using submodular optimization

This script selects a representative set of protein or DNA sequences from a larger set using submodular optimization. See [this manuscript](https://doi.org/10.1101/051201) for more information.



Required software:

* BLAST+ (https://blast.ncbi.nlm.nih.gov/Blast.cgi)
* BioPython (https://github.com/biopython/biopython.github.io/)
* path (https://pypi.python.org/pypi/forked-path)
* mysql-connector-python (https://pypi.org/project/mysql-connector-python/)

```
usage: repset.py [-h] --outdir OUTDIR --seqs SEQS [--mixture MIXTURE]

optional arguments:
  -h, --help         show this help message and exit
  --outdir OUTDIR    Output directory
  --seqs SEQS        Input sequences, fasta format
  --mixture MIXTURE  Mixture parameter determining the relative weight of
                         facility-location relative to sum-redundancy. Default=0.5
```

Output: Ordered list of sequence idenifiers, as defined in the input fasta file. The top N ids in this file represent the chosen representative set of size N.
