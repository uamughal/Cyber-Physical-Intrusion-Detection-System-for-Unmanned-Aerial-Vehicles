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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fusion", help="choose fusion")
parser.add_argument("--png", help="choose png name")
parser.add_argument("--metrics_title", help="print title name")
args = parser.parse_args()   # arg is variable


fusion = "central_cyber"
# fusion = args.fusion
df = pd.read_csv("../../dataset_updated/standardized_dataset_{}.csv".format(fusion))
df = df.drop(columns=['target'], axis=1)

keep_list = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
             'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
             'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble',
             'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof', 'target.value']  # 32

keep_list_dr_c = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                  'wlan.duration', 'wlan.seq', 'wlan.fc.type', 'wlan.flags', 'target.value']  # 12
keep_list_dr_cmb = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                    'wlan.duration', 'wlan.frag', 'wlan.seq', 'wlan.fc.type', 'wlan.flags', 'data.len', 'pitch', 'roll',
                    'templ', 'temph', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof', 'target.value']

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
X = df.iloc[:, :-1].values  # selecting all th rows and all the columns except the last one., i.e. all column except last
y = df.iloc[:, -1].values  # selecting only the last coulumn of the df
encoder = LabelEncoder()   # convert categorical data into integer values.
# For example, if your labels were ["cat", "dog", "cat", "bird"],
# LabelEncoder would transform them into something like [0, 1, 0, 2].
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
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape)) # original 32 filters in first layers
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # 64 filters in 2nd layers
    # model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',  metrics=['accuracy', 'Precision', 'Recall'])

    return model
# 'tanh', 'sigmoid', 'linear', or 'elu'.
# In the output layer, 'softmax' is common for multi-class
# classification, while 'sigmoid' is used for binary classification.
# Optimizer: 'Adam', 'SGD', 'RMSprop', or 'Adagrad'

def evaluate_model(model, X_test, y_test_one_hot, y_test_encoded):
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print('Classification Report')
    print(classification_report(y_test_encoded, y_pred))
    accuracy = accuracy_score(y_test_encoded, y_pred)
    recall = recall_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred)
    auc = roc_auc_score(y_test_encoded, y_pred)

    list_results = [accuracy, precision, recall, f1, auc]
    # print(f"{args.metrics_title}")  # --metrics_title= central_combined
    # print('accuracy, precision, recall, f1, auc :', list_results)
    # You can replace 'a' with 'w' if you want to overwrite the file each time instead of appending
    with open('results.txt', 'a') as f:
        f.write(f"{args.metrics_title}\n")  # --metrics_title= central_combined
        f.write('accuracy, precision, recall, f1, auc :' + str(list_results) + "\n")


def plot_curve(history):
    # F1 Score plot
    plt.figure(figsize=(14, 6))  # Modify the size here
    plt.grid()
    recall = np.array(history.history['recall'])
    precision = np.array(history.history['precision'])
    f1_train = (2 * ((precision * recall)/(precision + recall + 1e-7)))
    v_recall = np.array(history.history['val_recall'])
    v_precision = np.array(history.history['val_precision'])
    f1_val = (2 * ((v_precision * v_recall)/(v_precision + v_recall + 1e-7)))
    plt.plot(f1_train,  color='r', label='Training')
    plt.plot(f1_val, color='g', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epochs')
    plt.legend(loc='upper left')
    plt.savefig("cnn_results/F1_{}.png".format(fusion))
    plt.show()

    # Loss plot
    plt.figure(figsize=(14, 6))  # Modify the size here
    plt.grid()
    plt.plot(history.history['loss'],  color='r', label='Training')
    plt.plot(history.history['val_loss'], color='g', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.title('Loss vs. Epoch')
    plt.savefig("cnn_results/loss_{}.png".format(fusion))
    plt.show()


# def main():
    # fusion = "central_cyber"
    # X_train, y_train_one_hot, X_test, y_test_one_hot, y_test_encoded = load_and_preprocess_data(fusion)
model = create_model((X_train.shape[1], 1))
history = model.fit(X_train, y_train_one_hot, validation_split=0.2, epochs=50, batch_size=32, verbose=1)
evaluate_model(model, X_test, y_test_one_hot, y_test_encoded)
plot_curve(history)


# if __name__ == "__main__":
#    main()

# powershell command: python ID_CNN.py --fusion=drone1_cyber --metrics_title=drone1_cyber
