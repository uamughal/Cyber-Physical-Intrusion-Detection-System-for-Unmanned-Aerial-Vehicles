import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import shap
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose png name")
args = parser.parse_args()   # arg is variable


def create_model(neurons, act_f, hiddenlayers, ki):
    model = Sequential()
    model.add(LSTM(int(neurons), input_shape=(WS, X.shape[2]), return_sequences=True))

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

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

"""
def create_model(neurons, act_f, hiddenlayers, ki):
    model = Sequential()
    model.add(LSTM(int(neurons), input_shape=(WS, X.shape[2]), return_sequences=True))

    if hiddenlayers == 2:
        model.add(LSTM(int(neurons * 1 / 2), return_sequences=False))

    elif hiddenlayers == 3:
        model.add(LSTM(int(neurons * 2 / 3), kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(int(neurons * 1 / 3), kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL3'))

    elif hiddenlayers == 4:
        model.add(LSTM(int(neurons * 3 / 4), kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(int(neurons * 2 / 4), kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL3'))
        model.add(LSTM(int(neurons * 1 / 4), kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL4'))

    elif hiddenlayers == 5:
        model.add(LSTM(int(neurons * 4 / 5), kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(int(neurons * 3 / 5), kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL3'))
        model.add(LSTM(int(neurons * 2 / 5), kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL4'))
        model.add(LSTM(int(neurons * 1 / 5), kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL5'))

    model.add(Dense(1, activation='sigmoid', name='Output'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
"""
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
                                              baseline=None, restore_best_weights=True)

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
# keep_list_p = ['flight_time', 'agx', 'agy', 'agz', 'yaw', 'tof']


# fusion = "physical"   # fusion: combined, drone1, drone2, cyber, physical
fusion = args.fusion   # fusion: combined, drone1, drone2, cyber, physical
WS = 10   # window size
ki = 'he_uniform'

# X = np.load(path + "/data/Processed/{}_WS{}_X.npy".format(fusion, WS))
# Y = np.load(path + "/data/Processed/{}_WS{}_Y.npy".format(fusion, WS))
# XX = np.load(path + "/data/Processed/{}_WS{}_XX.npy".format(fusion, WS))
# YY = np.load(path + "/data/Processed/{}_WS{}_YY.npy".format(fusion, WS))


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


X, XX, Y, YY = [np.array(x) for x in train_test_split(xs, ys, shuffle=True)]

np.save("np_arrays/{}_X.npy".format(fusion), X)
np.save("np_arrays/{}_Y.npy".format(fusion), Y)
np.save("np_arrays/{}_XX.npy".format(fusion), XX)
np.save("np_arrays/{}_YY.npy".format(fusion), YY)

# Train the LSTM model
model = create_model(256, 'relu', 5, ki)
history = model.fit(x=X, y=Y, validation_data=(XX, YY), epochs=50, batch_size=128, callbacks=[early_stop]).history

# Create a background dataset for the SHAP explainer
background_data = X[np.random.choice(X.shape[0], 100, replace=False)]

# Initialize the explainer with the LSTM model and the background dataset
explainer = shap.DeepExplainer(model, background_data)

# Select a smaller dataset to explain (to save computation time)
X_sample = X[np.random.choice(X.shape[0], 50, replace=False)]

# Compute the SHAP values for the sample dataset
shap_values = explainer.shap_values(X_sample)

# Create the SHAP beeswarm plot
shap.summary_plot(shap_values[0], X_sample, plot_type="beeswarm", feature_names=keep_list)

# Save the SHAP beeswarm plot as a PNG file
plt.savefig("lstm_results/SHAP_beeswarm_{}_sam.png".format(fusion))
# plt.show()
plt.close()
