import os
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import KFold
import tensorflow as tf
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose png name")
parser.add_argument("--title_name", help="print title name")
args = parser.parse_args()   # arg is variable


def create_model(neurons, act_f, hiddenlayers, ki):
    model = Sequential()
    model.add(LSTM(int(neurons), input_shape=(WS, xs.shape[2]), return_sequences=True))

    if hiddenlayers == 2:
        model.add(LSTM(int(neurons * 1 / 2), return_sequences=False))

    elif hiddenlayers == 3:
        model.add(LSTM(int(neurons * 2 / 3), kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL2'))
        model.add(Dense(int(neurons * 1 / 3), activation=act_f))

    elif hiddenlayers == 4:
        model.add(LSTM(int(neurons * 3 / 4), kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(int(neurons * 2 / 4), kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL3'))
        model.add(Dense(int(neurons * 1 / 4), activation=act_f))

    elif hiddenlayers == 5:
        model.add(LSTM(int(neurons * 4 / 5), kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(int(neurons * 3 / 5), kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL3'))
        model.add(Dense(int(neurons * 2 / 5), activation=act_f))
        model.add(Dense(int(neurons * 1 / 5), activation=act_f))

    model.add(Dense(1, activation='sigmoid', name='Output'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=scores)

    return model


scores = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]

# path = os.getcwd()

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


# path = os.getcwd()
# fusion = "combined"   # fusion: combined, drone1, drone2, cyber, physical
fusion = args.fusion   # fusion: combined, drone1, drone2, cyber, physical

df = pd.read_csv("../../dataset_central/standardized_dataset_central_{}.csv".format(fusion))
df = df.set_index(['target'])
target = df['target.value']
df = df.drop(columns=['target.value'], axis=1)
# df = df.drop(drop_list_p, axis=1)

if fusion == "combined":
    df = df[keep_list]
elif fusion == "drone1":
    df = df[keep_list]
elif fusion == "drone2":
    df = df[keep_list]
elif fusion == "cyber":
    df = df[keep_list_c]
elif fusion == "physical":
    df = df[keep_list_p]

WS = 10   # window size
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

'''
np.save(path + "/data/lstm_{}_X.npy".format(fusion), xs)
np.save(path + "/data/lstm_{}_Y.npy".format(fusion), ys)

xs = np.load(path + "/data/thesis/{}_X.npy".format(fusion))
ys = np.load(path + "/data/thesis/{}_Y.npy".format(fusion))
'''
epochs = 100
splits = 5

kf = KFold(n_splits=splits, shuffle=True)
acc = 0
pre = 0
rec = 0
f1 = 0
auc = 0

ki = 'he_uniform'
for train, test in kf.split(xs, ys):

    model = create_model(256, 'relu', 5, ki)

    early_stop_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
                                                  baseline=None, restore_best_weights=True)
    early_stop_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
                                                    baseline=None, restore_best_weights=True)
    model.fit(x=xs[train], y=ys[train], callbacks=[early_stop_1], epochs=100, batch_size=64)

    loss, accuracy, precision, recall, auc_full = model.evaluate(xs[test], ys[test])
    acc += accuracy
    pre += precision
    rec += recall
    auc += auc_full
    f1 += (2 * ((precision * recall)/(precision + recall)))

acc = acc / splits
pre = pre / splits
rec = rec / splits
f1 = f1 / splits
auc = auc / splits

res = [acc, pre, rec, f1, auc]

# print(args.title_name)   # --title_name= combined
print(f"central_{fusion}")
print("Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}, AUC: {}".format(acc, pre, rec, f1, auc))
# print(res)
