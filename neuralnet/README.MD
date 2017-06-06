This repository is no longer maintained. It has been updated and cleaned to the last declassified/safe to release version which is from September 12, 2016.

Due to a switch to proprietary information, we are no longer updating the corpus, model, or python scripts.

Usage:

```python main.py path/to/dataset.csv #maxgpa #maxtestscore```

To restore checkpoints and train on one model, keep the dataset filename the same. To use a new model use a newly named dataset or delete the model from its directory under train/model/.

Note: with the Carnegie Mellon corpus and provided checkpoint (200 steps), relatively high accuracy can be achieved at 150000+ steps.
