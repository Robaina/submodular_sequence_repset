

# Activate environment
source /home/robaina/miniconda3/bin/activate
conda activate repset

# Run repset
python repset.py --seqs seqs.fasta --outdir test/

conda deactivate
