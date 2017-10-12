from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
from sys import argv
from console_logging.console import Console
console = Console()

usage="You shouldn't be running this file."

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
console.setVerbosity(3)  # only error, success, log

script = 'predict.py'
dataset_filename = './neuralnet/corpus/carnegie_mellon.csv'
maxgpa = 5.0
maxtest = 2400
dataset_filename = str(dataset_filename)
maxgpa = float(maxgpa)
maxtest = int(maxtest)
if dataset_filename[-4:] != ".csv":
    console.error("Filetype not recognized as CSV.")
    print(usage)
    exit(0)

# Data sets
DATA_TRAINING = dataset_filename
DATA_TEST = dataset_filename
''' We are expecting features that are floats (gpa, sat, act) and outcomes that are integers (0 for reject, 1 for accept) '''

# Load datasets using tf contrib libraries
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=DATA_TRAINING,
                                                                      target_dtype=np.int, features_dtype=np.float)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=DATA_TEST,
                                                                  target_dtype=np.int, features_dtype=np.float)
##

# First two columns are gpa, sat/act, which are our features
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]
##

# Build a neural network with 3 layers. We're putting the model into /train/model/
# I found 3 hidden layers with 10, 20, and 10 nodes respectively works well. You may find other setups.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="./neuralnet/train/model/" +
                                            dataset_filename[dataset_filename.rfind(
                                                '/') + 1:-4],
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=60))
##

# Helper functions
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y

def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y
##

maxsteps = 1
# Create the classifier. Take just one step, we're testing not training.
classifier.fit(input_fn=get_train_inputs, steps=maxsteps)

def predict(cur_gpa, testscore, test_type):
    #TODO: implement test_type
    gpa_in = cur_gpa
    testscore_in = testscore
    def new_samples():
        return np.array([[gpa_in, testscore_in]], dtype=np.float32)
    predictions = list(classifier.predict(input_fn=new_samples))
    return predictions