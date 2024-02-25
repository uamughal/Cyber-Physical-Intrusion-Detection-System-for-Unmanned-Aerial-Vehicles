import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

print('SVM, Area under the curve, standardized_dataset_central_cyber')

drop_list_c = ['timestamp_c', 'wlan_radio.signal_strength (dbm)', 'wlan_radio.Noise level (dbm)',
               'wlan_radio.SNR (db)', 'wlan_radio.preamble', 'wlan.frag', 'wlan.qos', 'wlan.qos.priority',
               'wlan.qos.ack', 'wlan.fcs.status', 'target.value']
drop_list_p = ['timestamp', 'mid', 'x', 'y', 'z', 'vgx', 'vgy', 'vgz', 'templ',
               'temph', 'baro', 'bat']

# Set the value of fusion parameter to "cyber", "physical", or "combined"
fusion = "cyber"

# Load the dataset from the CSV file
# df = pd.read_csv("../../dataset_central/dataset_{}.csv".format(fusion))
# df = pd.read_csv("../../dataset_central/previous_datasets/standardized_dataset_{}.csv".format(fusion))
df = pd.read_csv("../../dataset_central/old/full_dataset_standardized_{}.csv".format(fusion))

df['target'] = df['target']
df = df.drop(drop_list_c, axis=1)

# drop the target column
X = df.drop(['target'], axis='columns')   # target
y = df.target    # df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=209)
model = SVC(kernel='poly', C=5)   # model = SVC(kernel='poly', C=10, gamma=0.2)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print('Accuracy :', model.score(X_test, y_test))     # print(metrics.accuracy_score (y_test, y_pred=pred))
print(metrics.classification_report(y_test, y_pred=pred))

# Metrics
accuracy = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred, pos_label='attacked')
precision = precision_score(y_test, pred, pos_label='attacked')
f1 = f1_score(y_test, pred, pos_label='attacked')
auc = roc_auc_score(y_test, [1 if x == 'attacked' else 0 for x in pred])

print(fusion)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)
print("AUC:", auc)

"""
recall = recall_score(y_test, pred, pos_label='benign')
precision = precision_score(y_test, pred, pos_label='benign')
f1 = f1_score(y_test, pred, pos_label='benign')
auc = roc_auc_score(y_test, [1 if x == 'benign' else 0 for x in pred])
"""


