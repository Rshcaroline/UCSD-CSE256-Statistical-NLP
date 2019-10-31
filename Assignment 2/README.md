# [CSE 256 FA19] PA2: Semi-supervised Text Classification

## Files

There are several python files in the `root` folder:

You should be able to run supervised/semi-supervised model by:

```bash
python [semi-]sentiment[ wordvec].py
```

### Models

- `sentiment.py`: The supervised model for sentiment classification.
  - Features: BoW, TF-IDF
- `semi-sentiment.py`: The semi-supervised model for sentiment classification.
  - We use self-training algorithm to utilize the unlabeled data
- `semi-sentiment wordvec.py`: The semi-supervised model with word embedding trained on the unlabeled corpus.

### Supplementary files

- `classify.py`: This file help us to do classification and evaluation.
- `preprocess.py`: My preprocessing function.
- `mytokenize.py`: My tokenization function.

## Environment

You may need the following additional packages:

- Spacy
- Plotly
- Gensim