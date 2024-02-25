import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from tensorflow.keras.metrics import Precision, Recall
import ipywidgets as widgets
import warnings
import shap
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose png name")
parser.add_argument("--metrics_title", help="print title name")
args = parser.parse_args()   # arg is variable


fusion = args.fusion
# fusion = "central_cyber"
df = pd.read_csv("../../dataset_updated/standardized_dataset_{}.csv".format(fusion))
df = df.drop(columns=['target'], axis=1)

keep_list = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
             'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
             'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble',
             'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof', 'target.value']  # 32

keep_list_dr_c = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                  'wlan.duration', 'wlan.seq', 'wlan.fc.type', 'wlan.flags']  # 12
keep_list_dr_cmb = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                    'wlan.duration', 'wlan.frag', 'wlan.seq', 'wlan.fc.type', 'wlan.flags', 'data.len', 'pitch', 'roll',
                    'templ', 'temph', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']

keep_list_c = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
               'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
               'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble', 'target.value']  # 21
# remove cyber, if requires: wlan.fc.type, wlan.fcs.status, wlan_radio.channel, wlan_radio.preamble

keep_list_p = ['pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof', 'target.value']  # 11

# Select the columns to keep based on the fusion value
if fusion in ["drone1_cyber", "drone2_cyber", "central_cyber"]:
    df = df[keep_list_c]
elif fusion in ["drone1_physical", "drone2_physical", "central_physical"]:
    df = df[keep_list_p]
elif fusion in ["drone1_combined", "drone2_combined", "central_combined"]:
    df = df[keep_list]

# Preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Encode the target values
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# One-hot encode the target values
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

scores = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]


def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',  metrics=['accuracy', 'Precision', 'Recall'])

    return model


# fusion = "central_cyber"
# X_train, y_train_one_hot, X_test, y_test_one_hot, y_test_encoded = load_and_preprocess_data(fusion)
model = create_model((X_train.shape[1], 1))
history = model.fit(X_train, y_train_one_hot, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Transform your model to a function to use with SHAP
# Make sure the data passed into the model is in the correct shape
f = lambda x: model.predict(x.reshape(-1, X_train.shape[1], 1))

s_size = 150  # sample data
# Now you can use the 2D data with the explainer
data = shap.sample(X_train.reshape(X_train.shape[0], -1), s_size)  # reshape to 2D
data2 = shap.sample(X_test.reshape(X_test.shape[0], -1), s_size)  # reshape to 2D
# print(data.shape, data2.shape)

# Create a SHAP explainer
explainer = shap.KernelExplainer(f, data)
# explainer = shap.DeepExplainer(f, data)
# Calculate SHAP values
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(data2)

list_of_labels = df.columns.to_list()
tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))

# Create a widget for the labels and then display the widget
current_label = widgets.Dropdown(options=tuple_of_labels, value=0, description='Select Label:')

# Plot SHAP values
shap.summary_plot(shap_values[current_label.value], features=data2, feature_names=df.columns, show=False)
# shap.summary_plot(shap_values=shap_values[current_label.value], features=data2, feature_names=list_of_labels, show=False)
# If you want to generate the plot immediately for a specific feature (e.g., feature 0), you could do:
# shap.summary_plot(shap_values[0], features=data2, feature_names=df.columns, plot_type="beeswarm")

path_results = "cnn_results/"
# plt.savefig(path_results + args.png)
plt.savefig("cnn_results/SHAP_{}.png".format(fusion))
# plt.show()

plt.close()

# plt.savefig("cnn_results/shap_{}.png".format(fusion))
# plt.show()
# plt.close()
# path = 'svm_results/'
# plt.savefig(path + args.png)


# powershell command: python ID_CNN_shap.py --fusion=drone1_cyber
#
# --metrics_title=drone1_cyber
