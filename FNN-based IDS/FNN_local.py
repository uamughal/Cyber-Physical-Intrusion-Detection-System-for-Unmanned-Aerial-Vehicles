import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose png name")
parser.add_argument("--title_name", help="print title name")
args = parser.parse_args()   # arg is variable


def create_model(neurons, act_f, hiddenlayers):
    model = Sequential()

    if hiddenlayers == 2:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(neurons, activation=act_f))

    elif hiddenlayers == 3:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))

    elif hiddenlayers == 4:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))

    elif hiddenlayers == 5:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))

    elif hiddenlayers == 6:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))
        model.add(Dense(neurons, activation=act_f))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=scores)

    return model


keep_list = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
             'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
             'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble',
             'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']  # 32

keep_list_dr_c = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                  'wlan.duration', 'wlan.seq', 'wlan.fc.type', 'wlan.flags']  # 12
keep_list_dr_cmb = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                    'wlan.duration', 'wlan.frag', 'wlan.seq', 'wlan.fc.type', 'wlan.flags', 'data.len', 'pitch', 'roll',
                    'templ', 'temph', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']

keep_list_cn_c = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
                  'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
                  'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble']  # 21
# remove cyber, if requires: wlan.fc.type, wlan.fcs.status, wlan_radio.channel, wlan_radio.preamble

keep_list_p = ['pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']  # 11

# fusion = drone1_cyber, drone1_physical, drone1_combined, ... central_cyber, central_physical, central_combined
# fusion = args.fusion
fusion = 'drone1_cyber'
fusion_print = 'drone1_combined'

df = pd.read_csv("../../dataset_central/dataset_updated/standardized_dataset_{}.csv".format(fusion))
# df = pd.read_csv("../../dataset_central/contatenate_df/concatenated_{}.csv".format(fusion))
df = df.set_index(['target'])
target = df['target.value']
df = df.drop(columns=['target.value'], axis=1)

if fusion in ["drone1_cyber", "drone2_cyber", "central_cyber"]:
    df = df[keep_list_cn_c]
elif fusion in ["drone1_physical", "drone2_physical", "central_physical"]:
    df = df[keep_list_p]
elif fusion in ["drone1_combined", "drone2_combined", "central_combined"]:
    # df = df[keep_list_dr_cmb]
    df = df[keep_list]

# S = True
X, XX, Y, YY = [np.array(x) for x in train_test_split(df.values, target.values)]


X = np.vstack((X, XX))
Y = np.hstack((Y, YY))

scores = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]

kf = KFold(n_splits=5, shuffle=True)
fin_f1 = []
fin_val_f1 = []
fin_loss = []
fin_val_loss = []
i = 0
itera = ''

sacc = 0
spre = 0
srec = 0
sf1 = 0
sauc = 0
best_epochs = []  # to store the best epoch for each fold
# activation function: relu, sigmoid
for train, test in kf.split(X, Y):
    if i != 0:
        itera = '_' + str(i)

    model = create_model(32, 'tanh', 3)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss'.format(itera), min_delta=1e-2, patience=5, verbose=0, mode='auto',
                                                  baseline=None, restore_best_weights=True)
    history = model.fit(x=X[train], y=Y[train], validation_data=(X[test], Y[test]), callbacks=[early_stop], epochs=100, batch_size=400).history

    # Save the epoch at which early stopping occurred
    best_epochs.append(len(history['loss']))
    recall = np.array(history['recall'])
    precision = np.array(history['precision'])
    f1 = (2 * ((precision * recall)/(precision + recall)))

    v_recall = np.array(history['val_recall'])
    v_precision = np.array(history['val_precision'])
    v_f1 = (2 * ((v_precision * v_recall)/(v_precision + v_recall)))

    fin_loss.append(history['loss'])
    fin_val_loss.append(history['val_loss'])

    fin_f1.append(f1.tolist())
    fin_val_f1.append(v_f1.tolist())

    sacc += history['val_accuracy'][len(history['val_accuracy']) - 1]
    spre += history['val_precision'][len(history['val_precision']) - 1]
    srec += history['val_recall'][len(history['val_recall']) - 1]
    sf1 += v_f1[v_f1.shape[0] - 1]
    sauc += history['val_auc'][len(history['val_auc']) - 1]

    i += 1


length = max(map(len, fin_f1))
y = np.array([xi+[np.NaN]*(length-len(xi)) for xi in fin_f1])

length = max(map(len, fin_val_f1))
y2 = np.array([xi+[np.NaN]*(length-len(xi)) for xi in fin_val_f1])

length = max(map(len, fin_loss))
y3 = np.array([xi+[np.NaN]*(length-len(xi)) for xi in fin_loss])

length = max(map(len, fin_val_loss))
y4 = np.array([xi+[np.NaN]*(length-len(xi)) for xi in fin_val_loss])


train_scores_mean = np.nanmean(y, axis=0)
train_scores_std = np.nanstd(y, axis=0)
test_scores_mean = np.nanmean(y2, axis=0)
test_scores_std = np.nanstd(y2, axis=0)

x_ticks = np.arange(length)
title = r"Learning Curves (FNN, 5 layers, 256 neurons, relu, glorot)"
_, axes = plt.subplots(1, 1, figsize=(14, 6))
axes.grid()
axes.fill_between(
    x_ticks,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",
)
axes.fill_between(
    x_ticks,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g",
)
path_results = 'fnn_results/'

# axes.plot(x_ticks, train_scores_mean, 'o-', color="r", label='Train', linewidth=3)
# axes.plot(x_ticks, test_scores_mean, 'o-', color="g", label='Validation', linewidth=3)
axes.plot(x_ticks, train_scores_mean, 'o-', color="r", label='Train')
axes.plot(x_ticks, test_scores_mean, 'o-', color="g", label='Validation')
axes.set_ylabel('F1')
axes.set_xlabel('Epoch')
axes.legend(loc='upper left')
# plt.xlim(0, 17)
# plt.savefig(path_results + args.png)
plt.savefig("fnn_results/F1_{}.png".format(fusion_print))
# plt.show()


train_scores_mean = np.nanmean(y3, axis=0)
train_scores_std = np.nanstd(y3, axis=0)
test_scores_mean = np.nanmean(y4, axis=0)
test_scores_std = np.nanstd(y4, axis=0)

x_ticks = np.arange(length)
# title = r"Learning Curves (FNN, 5 layers, 256 neurons, relu, glorot)"
_, axes = plt.subplots(1, 1, figsize=(14, 6))
axes.grid()
axes.fill_between(
    x_ticks,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",
)
axes.fill_between(
    x_ticks,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g",
)

# line width
# axes.plot(x_ticks, train_scores_mean, 'o-', color="r", label='Train', linewidth=3)
# axes.plot(x_ticks, test_scores_mean, 'o-', color="g", label='Validation', linewidth=3)
axes.plot(x_ticks, train_scores_mean, 'o-', color="r", label='Train')
axes.plot(x_ticks, test_scores_mean, 'o-', color="g", label='Validation')
axes.set_ylabel('Loss')
axes.set_xlabel('Epoch')
axes.legend(loc='upper left')
# plt.xlim(0, 17)
# plt.savefig(path_results + args.png)
plt.savefig("fnn_results/loss_{}.png".format(fusion_print))
# plt.show()

sacc = sacc / 5
spre = spre / 5
srec = srec / 5
sf1 = sf1 / 5
sauc = sauc / 5

list_results1 = [sacc, spre, srec, sf1, sauc]
# print(args.title_name)
print(fusion_print)
print('accuracy, precision, recall, f1, auc :', list_results1)

# powershell prompt: python FNN_f.py --fusion=drone1_cyber --title_name=drone1_cyber
