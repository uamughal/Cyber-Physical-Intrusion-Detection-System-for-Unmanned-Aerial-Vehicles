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
import shap
import ipywidgets as widgets
import warnings
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose png name")
parser.add_argument("--title_name", help="print title name")
args = parser.parse_args()   # arg is variable
shap.initjs()


def create_model(neurons, act_f, hiddenlayers):
    model = Sequential()

    if hiddenlayers == 2:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(int(neurons * 1 / 2), activation=act_f))

    elif hiddenlayers == 3:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(int(neurons * 2 / 3), activation=act_f))
        model.add(Dense(int(neurons * 1 / 3), activation=act_f))

    elif hiddenlayers == 4:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(int(neurons * 3 / 4), activation=act_f))
        model.add(Dense(int(neurons * 2 / 4), activation=act_f))
        model.add(Dense(int(neurons * 1 / 4), activation=act_f))

    elif hiddenlayers == 5:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(int(neurons * 4 / 5), activation=act_f))
        model.add(Dense(int(neurons * 3 / 5), activation=act_f))
        model.add(Dense(int(neurons * 2 / 5), activation=act_f))
        model.add(Dense(int(neurons * 1 / 5), activation=act_f))

    elif hiddenlayers == 6:
        model.add(Dense(neurons, input_shape=(X.shape[1],), activation=act_f))
        model.add(Dense(int(neurons * 5 / 6), activation=act_f))
        model.add(Dense(int(neurons * 4 / 6), activation=act_f))
        model.add(Dense(int(neurons * 3 / 6), activation=act_f))
        model.add(Dense(int(neurons * 2 / 6), activation=act_f))
        model.add(Dense(int(neurons * 1 / 6), activation=act_f))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


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


# fusion = "cyber"
fusion = args.fusion   # fusion: combined, drone1, drone2, cyber, physical

df = pd.read_csv("../../dataset_central/standardized_dataset_central_{}.csv".format(fusion))
df = df.set_index(['target'])
target = df['target.value']
df = df.drop(columns=['target.value'], axis=1)

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

list_of_labels = df.columns.to_list()

tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))

# Create a widget for the labels and then display the widget
current_label = widgets.Dropdown(options=tuple_of_labels, value=0, description='Select Label:')

S = True
X, XX, Y, YY = [np.array(x) for x in train_test_split(df.values, target.values, shuffle=S)]


model = create_model(256, 'relu', 6)
history = model.fit(x=X, y=Y, validation_data=(XX, YY), epochs=5, batch_size=64).history

s_size = 150  # sample data
data = shap.sample(X, s_size)
data2 = shap.sample(XX, s_size)

explainer = shap.KernelExplainer(model.predict, data)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    shap_values = explainer.shap_values(data2)
# shap.force_plot(explainer.expected_value[0], shap_values[0], data2, matplotlib=True)

# title = "FNN -- Combined Data"

# shap.summary_plot(shap_values=shap_values[current_label.value], features=X[0:s_size, :], feature_names=list_of_labels, title=title, show=False)
shap.summary_plot(shap_values=shap_values[current_label.value], features=X[0:s_size, :], feature_names=list_of_labels, show=False)
# plt.title(title)
# plt.show()
path_results = "fnn_results/"
# plt.savefig(path_results + args.png)
plt.savefig("fnn_results/FNN_SHAP_{}.png".format(fusion))
# plt.show()
plt.close()
# python fnn_shap_f.py --fusion=drone2 --png=drone2_shap.png
