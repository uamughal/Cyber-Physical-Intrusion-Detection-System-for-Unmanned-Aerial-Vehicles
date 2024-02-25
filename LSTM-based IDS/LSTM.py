import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from keras.layers import Dropout
from keras.regularizers import L1, L2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose png name")
parser.add_argument("--metrics_title", help="print title name")
args = parser.parse_args()   # arg is variable


def create_model(neurons, act_f, hiddenlayers, ki, l1_value, l2_value, dropout_rate, learning_rate):
    model = Sequential()
    # Add L1 and L2 regularizers to the LSTM layer
    lstm_regularizer = L1(l1_value) if l1_value > 0 else L2(l2_value)
    model.add(LSTM(int(neurons), input_shape=(WS, X.shape[2]), return_sequences=True, kernel_regularizer=lstm_regularizer))

    if hiddenlayers == 2:
        model.add(LSTM(neurons, return_sequences=False))

    elif hiddenlayers == 3:
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL3'))

    elif hiddenlayers == 4:
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL3'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL4'))

    elif hiddenlayers == 5:
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL3'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL4'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL5'))

    # Add dropout layer after each LSTM layer
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid', name='Output'))
    optimizer = Adam(lr=learning_rate)  # default learning rate for adam optimizer is 0.001

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=scores)

    return model


scores = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]

"""
# Define the list of columns to be dropped from the dataset
drop_list_c1 = ['frame.time_epoch', 'wlan_radio.signal_strength (dbm)', 'wlan_radio.Noise level (dbm)',
                'wlan_radio.SNR (db)', 'wlan_radio.preamble', 'wlan.frag', 'wlan.qos', 'wlan.qos.priority',
                'wlan.qos.ack', 'wlan.fcs.status']
drop_list_c2 = ['wlan.bssid', 'wlan.ta', 'wlan.wep.key', 'radiotap.antenna_signal', 'radiotap.channel.flags.cck',
                'wlan_radio.channel', 'wlan_radio.frequency', 'frame.number', 'radiotap.channel.flags.ofdm',
                'wlan.fc.type']  # 20/36 removed
drop_list_p = ['timestamp', 'mid', 'x', 'y', 'z', 'vgx', 'vgy', 'vgz', 'templ',
               'temph', 'baro', 'bat', 'pitch', 'roll', 'h', ]   # 15/21 removed
# 20 keep (combined)
keep_list = ['frame.len', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.duration', 'wlan.seq', 'wlan.fc.subtype',
             'wlan.flags', 'wlan.fcs', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length', 'radiotap.signal_quality',
             'wlan_radio.datarate', 'flight_time', 'agx', 'agy', 'agz', 'yaw', 'tof']
keep_list_c = ['frame.len', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.duration', 'wlan.seq', 'wlan.fc.subtype',
               'wlan.flags', 'wlan.fcs', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length', 'radiotap.signal_quality',
               'wlan_radio.datarate']
keep_list_p = ['flight_time', 'agx', 'agy', 'agz', 'yaw', 'tof', 'pitch', 'vgz']
"""
keep_list = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
             'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
             'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble',
             'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']  # 32

keep_list_dr_c = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                  'wlan.duration', 'wlan.seq', 'wlan.fc.type', 'wlan.flags']  # 12
keep_list_dr_cmb = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                    'wlan.duration', 'wlan.frag', 'wlan.seq', 'wlan.fc.type', 'wlan.flags', 'data.len', 'pitch', 'roll',
                    'templ', 'temph', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']

keep_list_c = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
               'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
               'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble']  # 21
# remove cyber, if requires: wlan.fc.type, wlan.fcs.status, wlan_radio.channel, wlan_radio.preamble

keep_list_p = ['pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']  # 11


# path = os.getcwd()
# fusion = drone1_cyber, drone1_physical, drone1_combined, ... central_cyber, central_physical, central_combined
fusion = args.fusion # cyber
WS = 10   # window size

"""
df = pd.read_csv("../../dataset_updated/standardized_dataset_{}.csv".format(fusion))
df = df.set_index(['target'])
target = df['target.value']
df = df.drop(columns=['target.value'], axis=1)
# df = df.drop(drop_list_p, axis=1)

if fusion in ["drone1_cyber", "drone2_cyber", "central_cyber"]:
    df = df[keep_list_c]
elif fusion in ["drone1_physical", "drone2_physical", "central_physical"]:
    df = df[keep_list_p]
elif fusion in ["drone1_combined", "drone2_combined", "central_combined"]:
    # df = df[keep_list_dr_cmb]
    df = df[keep_list]


feats = df.shape[1]
xs = np.empty((1, WS, df.shape[1]))
ys = np.empty(1)

for i in range(0, (df.values.shape[0] - WS)):
    if i == 0:
        xs = np.reshape(df.values[i:(i + WS)], (1, WS, feats))
        ys = target.values[i+WS]
    else:
        temp_arr = np.reshape(df.values[i:(i + WS)], (1, WS, feats))
        xs = np.vstack((xs, temp_arr))
        ys = np.hstack((ys, target.values[i+WS]))

X = xs
Y = ys
# X, XX, Y, YY = [np.array(x) for x in train_test_split(xs, ys, shuffle=True)]
# X = np.vstack((X, XX))
# Y = np.hstack((Y, YY))

np.save("np_arrays/{}_X.npy".format(fusion), X)
np.save("np_arrays/{}_Y.npy".format(fusion), Y)
"""

X = np.load("np_arrays/{}_X.npy".format(fusion))
Y = np.load("np_arrays/{}_Y.npy".format(fusion))

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

# lecun_uniform, lecun_normal, zeros, ones, orthogonal, identity, constant )
act_f = 'relu'    # tanh, sigmoid, softmax,
ki = 'he_uniform'  # kernel initialization (he_uniform, glorot_uniform, glorot_normal,
l1_value = 0.0001   # 0.0001
l2_value = 0.001   # 0.001
dropout_rate = 0.7  # 0.8
learning_rate = 0.0001  # default learning_rate for adam optimizer= 0.001 # 0.0001

for train, test in kf.split(X, Y):
    if i != 0:
        itera = '_' + str(i)

    model = create_model(neurons=512, act_f=act_f, hiddenlayers=5, ki=ki, l1_value=l1_value, l2_value=l2_value,
                         dropout_rate=dropout_rate, learning_rate=learning_rate)

    early_stop = EarlyStopping(monitor='val_loss'.format(itera), min_delta=1e-2, patience=5, verbose=1, mode='auto',
                                                  baseline=None, restore_best_weights=True)
    history = model.fit(x=X[train], y=Y[train], validation_data=(X[test], Y[test]), callbacks=[early_stop], epochs=100, batch_size=128).history

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
title = r"Learning Curves (5 layers, 256 neurons, relu, glorot)"
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

# axes.plot(x_ticks, train_scores_mean, 'o-', color='r', label='Train', linewidth=3)  # linewidth
# axes.plot(x_ticks, test_scores_mean, 'o-', color='g', label='Validation', linewidth=3)
axes.plot(x_ticks, train_scores_mean, 'o-', color='r', label='Train')
axes.plot(x_ticks, test_scores_mean, 'o-', color='g', label='Validation')
axes.set_ylabel('F1')
axes.set_xlabel('Epoch')
axes.legend(loc='upper left')
# plt.xlim(0, 17)
# plt.savefig(path + "/Figures/FINAL/lstm_{}_F1.png".format(fusion))
# plt.savefig("lstm_results/lstm_F1_reg_{}".format(args.png))
plt.savefig("lstm_results/F1_{}.png".format(fusion))
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

axes.plot(x_ticks, train_scores_mean, 'o-', color="r", label='Train')
axes.plot(x_ticks, test_scores_mean, 'o-', color="g", label='Validation')
axes.set_ylabel('Loss')
axes.set_xlabel('Epoch')
axes.legend(loc='upper left')
# plt.xlim(0, 17)
# plt.savefig(path + "/Figures/FINAL/lstm_{}_F1.png".format(fusion))
plt.savefig("lstm_results/loss_{}.png".format(fusion))
# plt.savefig("lstm_results/loss_{}".format(args.png))
# plt.show()

sacc = sacc / 5
spre = spre / 5
srec = srec / 5
sf1 = sf1 / 5
sauc = sauc / 5

list_results = [sacc, spre, srec, sf1, sauc]

"""
# You can replace 'a' with 'w' if you want to overwrite the file each time instead of appending
with open('results_5.txt', 'a') as f:
    f.write(f"{args.metrics_title}\n")  # --metrics_title= central_combined
    f.write('accuracy, precision, recall, f1, auc :' + str(list_results) + "\n")
"""
# print(args.metrics_title)  # --metrics_title= central_combined
print(f"central_{args.metrics_title}")  # --metrics_title= central_combined
print('accuracy, precision, recall, f1, auc :', list_results)
# print(list_results)


# powershell command: python LSTM.py --fusion=drone1_cyber --metrics_title=drone1_cyber
