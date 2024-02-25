# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose .png name")
parser.add_argument("--metrics_title", help="print metrics title")
args = parser.parse_args()   # arg is variable

print('Hyperparameters: kernel: poly, C:10, gamma:0.2')


def plot_learning_curve(estimator, X, y, num_training=None, axes=None, ylim=None, cv=None,
                        n_jobs=1, scoring=None):
    """
    Generate 3 plots: the learning curve, the training set scores, the test set scores
    """
    # If no axes object is provided, create one
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(14, 6))

    # Set the label of x-axis
    axes.set_xlabel("Training examples")

    # Set the label of y-axis
    axes.set_ylabel("F1 Score")

    # Set the range of y-axis
    if ylim is not None:
        axes.set_ylim(*ylim)

    # Specify the sizes of the training sets
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, num_training), return_times=True)

    # Print out the results
    # print('np.linspace', np.linspace(0.1, 1.0, num_training))
    # print('train_sizes:', train_sizes)
    # print('train_scores:', train_scores)
    # print('test_scores:', test_scores)

    # Calculate the mean and standard deviation of the training scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    # Add a legend to the plot
    axes.legend(loc="best")
    print('train_scores_mean :', train_scores_mean)
    print('test_scores_mean :', test_scores_mean)

    return plt


# timestamp_p, timestamp_c
keep_list = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
             'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
             'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble',
             'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']  # 32

keep_list_c = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
               'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
               'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble']  # 21
# remove cyber, if requires: wlan.fc.type, wlan.fcs.status, wlan_radio.channel, wlan_radio.preamble

keep_list_p = ['pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']  # 11


# fusion = drone1_cyber, drone1_physical, drone1_combined, ... central_cyber, central_physical, central_combined
fusion = args.fusion
df = pd.read_csv("../../dataset_updated/standardized_dataset_{}.csv".format(fusion))
df = df.set_index(['target'])
target = df['target.value']
df = df.drop(columns=['target.value'], axis=1)

if fusion in ["drone1_cyber", "drone2_cyber", "central_cyber"]:
    df = df[keep_list_c]
elif fusion in ["drone1_physical", "drone2_physical", "central_physical"]:
    df = df[keep_list_p]
elif fusion in ["drone1_combined", "drone2_combined", "central_combined"]:
    df = df[keep_list]

S = True   # shuffle
# Split the dataset into training and testing sets
X, XX, Y, YY = [np.array(x) for x in train_test_split(df.values, target.values, shuffle=S)]

# X = np.vstack((X, XX))
# Y = np.hstack((Y, YY))

scores = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
# kernel: rbf, poly, linear, sigmoid
# Create a figure to plot the learning curve
# Create a figure to plot the learning curve
fig, axes = plt.subplots(1, 1, figsize=(14, 6))
model = SVC(kernel='poly', C=5, gamma=0.1)  # C=10, gamma=0.2 regularization parameter

# Plot the learning curve for the SVM classifier with 10-fold cross-validation using F1 score as evaluation metric
plot_learning_curve(estimator=model, X=X, y=Y, num_training=10, cv=5, scoring='f1')

path = 'svm_results/'
plt.savefig(path + args.png)  # save the plot

# Performance Metrics
results = cross_validate(model, X, Y, cv=5, scoring=scores)  # cross-validation, cv=5

acc = np.average(results['test_accuracy'])
pre = np.average(results['test_precision'])
rec = np.average(results['test_recall'])
f1 = np.average(results['test_f1'])
auc = np.average(results['test_roc_auc'])

res = [acc, pre, rec, f1, auc]

# choose print name as an argument
print(args.metrics_title)
# print(metrics_title)
print('accuracy, precision, recall, f1, auc :', res)


# powershell command: python svm_both.py --fusion=drone1_cyber --png=drone1_cyber.png --metrics_title=drone1_cyber
