python replace_with_rare.py gene.train > gene.train_with_rare
python count_freqs.py gene.train_with_rare > gene.counts_with_rare
python viterbi.py gene.counts_with_rare gene.dev > gene_dev.p3.out
python eval_gene_tagger.py gene.key gene_dev.p3.out
