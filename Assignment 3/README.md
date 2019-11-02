# [CSE 256 FA19] PA3: Sequence Tagging 

## Part 1. Baseline

### Problem 1. 

Files are in `root/P1_Baseline/p1`. You can simple run:

```bash
bash p1-1.sh
```

Or following the next steps:

#### Pre-processing

> Replace infrequent words (Count(x) < 5) in the original training data file with a common symbol `_RARE_` . Then re-run `count_freqs.py` to produce new counts. 

```bash
python replace_with_rare.py gene.train > gene.train_with_rare
python count_freqs.py gene.train_with_rare > gene.counts_with_rare
```

#### Training and Evaluating

> As a baseline, implement a simple gene tagger that always produces the tag $y^{*}=\arg \max _{y} e(x | y)$ for each word $x$. Make sure your tagger uses the `_RARE_` word probabilities for rare and unseen words. Your tagger should read in the counts file and the file `gene.dev` (which is `gene.key` without the tags) and produce output in the same format as the training file. 

```bash
python baseline.py gene.counts_with_rare gene.dev > gene_dev.p1.out
python eval_gene_tagger.py gene.key gene_dev.p1.out
```

```bash
Found 2669 GENEs. Expected 642 GENEs; Correct: 424.

         precision      recall          F1-Score
GENE:    0.158861       0.660436        0.256116
```

### Problem 2.

Files are in `root/P1_Baseline/p2`. You can simple run:

```bash
bash p1-2.sh
```

Or following the next steps:

#### Pre-processing


> Your tagger can be improved by grouping words into informative word classes rather than just into a single class of rare and unseen words. Propose more informative word classes for dealing with such words. 

```bash
python replace_with_rare_classes.py gene.train > gene.train_with_rare_classes
python count_freqs.py gene.train_with_rare_classes > gene.counts_with_rare_classes
```

#### Training and Evaluating 

```bash
python baseline.py gene.counts_with_rare_classes gene.dev > gene_dev.p2.out
python eval_gene_tagger.py gene.key gene_dev.p2.out
```

```bash
Found 2109 GENEs. Expected 642 GENEs; Correct: 400.

         precision      recall          F1-Score
GENE:    0.189663       0.623053        0.290803
```

## Part 2. Trigram HMM

