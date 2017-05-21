from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
from sys import argv
usage = "\nUsage:\npython main.py path/to/dataset.csv #MAX_GPA #MAX_TEST_SCORE\n\nExample:\tpython main.py harvard.csv 6.0 2400\n\nThe dataset should have one column of GPA and one column of applicable test scores, no headers."
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    script, dataset_filename, maxgpa, maxtest = argv
except:
    print("An error has occurred: "+str(sys.exc_info()[0]))
    print(usage)
    exit(0)
dataset_filename = str(dataset_filename)
maxgpa = float(maxgpa)
maxtest = int(maxtest)
if dataset_filename[-4:] != ".csv":
    print("The file extension is not .csv - error.")
    print(usage)
    exit(0)
# Data sets
DATA_TRAINING = dataset_filename
DATA_TEST = dataset_filename
''' We are expecting features that are floats (gpa, sat, act) and outcomes that are integers (0 for reject, 1 for accept) '''

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=DATA_TRAINING,
                                                       target_dtype=np.int,features_dtype=np.float)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=DATA_TEST,
                                                   target_dtype=np.int,features_dtype=np.float)

# First two columns are gpa, sat/act, which are our features
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

# Build a neural network with 3 layers. We're putting the model into /train/model/
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="./train/model/"+dataset_filename[dataset_filename.rfind('/')+1:-4],
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=60))

def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y

def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y

print("How many steps should we train for?")
maxsteps = int(input('> '))
# Create the classifier. Take maxsteps steps.
classifier.fit(input_fn=get_train_inputs, steps=maxsteps)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
print('\nFinished with accuracy {0:f}'.format(accuracy_score))

print("Please provide a GPA and test score to chance.")
cur_gpa = float(input('GPA: '))
test_score = int(input('Test Score: '))
def new_samples():
    return np.array([[0.0, 0], [cur_gpa,test_score], [maxgpa, maxtest]], dtype=np.float32)
predictions = list(classifier.predict(input_fn=new_samples))
print("Made predictions.")

def returnChance(chance):
    if chance==0:
        return "rejection"
    if chance==1:
        return "admission"

print("For GPA 0 and test score 0, the result will be "+returnChance(predictions[0]))
print("For GPA "+str(cur_gpa)+" and test score "+str(test_score)+", the result will be "+returnChance(predictions[1]))
print("For GPA "+str(maxgpa)+" and test score "+str(maxtest)+", the result will be "+returnChance(predictions[2]))