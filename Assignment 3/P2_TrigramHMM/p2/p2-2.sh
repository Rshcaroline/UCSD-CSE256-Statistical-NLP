python replace_with_rare_classes.py gene.train > gene.train_with_rare_classes
python count_freqs.py gene.train_with_rare_classes > gene.counts_with_rare_classes
python viterbi.py gene.counts_with_rare_classes gene.dev > gene_dev.p4.out
python eval_gene_tagger.py gene.key gene_dev.p4.out
