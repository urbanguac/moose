This repository is no longer maintained. It has been updated and cleaned to the last declassified/safe to release version which is from September 12, 2016.

Due to a switch to proprietary information, we are no longer updating the corpus, model, or python scripts.

Usage:

```python neuralnet/main.py path/to/dataset.csv #maxgpa #maxtestscore```

Run from main directory.

To restore checkpoints and train on one model, keep the dataset filename the same. To use a new model use a newly named dataset or delete the model from its directory under train/model/.

Note: with the Carnegie Mellon corpus and provided checkpoint (150000 steps), relatively high accuracy can be achieved upon further training.

Training stats:
Trained on GeForce 1060, 6GB. ~4 minutes for 150,000 steps @ 78.5% accuracy on a relatively small dataset.

No overfitting observed. Cross validation dataset and cross validation scripts are not provided due to licensing issues. :sadparrot:

*Interestingly, it would appear even if overfitting occurred, that this may not affect our true accuracy, as college admissions behave as if overfitted... but this is a conjecture to be tested at a later point.*