# [CSE 256 FA19] PA4: Machine Translation

## Part 1. IBM Model 1

```bash
python IBM1.py
python eval_alignment.py ../data/dev.key alignment.p1.out
```

```bash
      Type       Total   Precision      Recall     F1-Score
===============================================================
     total        5920     0.413        0.427        0.420
```

## Part 2. IBM Model 2

```
python IBM2.py
python eval_alignment.py ../data/dev.key alignment.p2.out
```

```bash
      Type       Total   Precision      Recall     F1-Score
===============================================================
     total        5920     0.442        0.456        0.449
```

## Part 3. Growing Alignments

```
python rev_IBM2.py
python grow_alignment.py
python eval_alignment.py ../data/dev.key and_alignment.out
```

```bash
      Type       Total   Precision      Recall     F1-Score
===============================================================
     total        5920     0.823        0.270        0.407
```

```
python eval_alignment.py ../data/dev.key or_alignment
```

```bash
      Type       Total   Precision      Recall     F1-Score
===============================================================
     total        5920     0.320        0.538        0.401
```

```
python eval_alignment.py ../data/dev.key alignment.p3.out
```

```bash
      Type       Total   Precision      Recall     F1-Score
===============================================================
     total        5920     0.662        0.369        0.474
```

