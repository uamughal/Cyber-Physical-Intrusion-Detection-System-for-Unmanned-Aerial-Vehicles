import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


path = os.getcwd()

# drop_list = ['timestamp_p', 'timestamp_c', 'battery', 'temperature', 'frame.number', 'barometer']
keep_list = ['frame.len', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.duration', 'wlan.seq', 'wlan.fc.subtype',
             'wlan.flags', 'wlan.fcs', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length', 'radiotap.signal_quality',
             'wlan_radio.datarate', 'time', 'agx', 'agy', 'agz', 'yaw', 'tof']

keep_list_c = ['frame.len', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.duration', 'wlan.seq', 'wlan.fc.subtype',
               'wlan.flags', 'wlan.fcs', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length', 'radiotap.signal_quality',
               'wlan_radio.datarate']

keep_list_p = ['time', 'agx', 'agy', 'agz', 'yaw', 'tof', 'pitch', 'vgz']

epochs = 20


S = True
fusion = "combined"


# df = pd.read_csv(path + "/data/Standardized/full_dataset_standardized_combined.csv".format(fusion))
df = pd.read_csv("../../dataset_central/dataset_central_{}.csv".format(fusion))
# df = pd.read_csv("../dataset_central/standardized_dataset_central_{}.csv".format(fusion))
df = df.set_index(['target'])
target = df['target.value']
df = df.drop(columns=['target.value'], axis=1)

if fusion == "combined":
    df = df[keep_list]
elif fusion == "cyber":
    df = df[keep_list_c]
elif fusion == "physical":
    df = df[keep_list_p]

# df = df.drop(columns=drop_list, axis=1)
X, XX, Y, YY = [np.array(x) for x in train_test_split(df.values, target.values, shuffle=S)]
feats = X.shape[1]
estimator = SVC()
scale = 1 / (feats * X.var())
kernel = ['linear', 'rbf', 'poly', 'sigmoid']
C = [1, 10, 100]
gamma = [0.2, 0.3, 0.5]
param_grid = dict(C=C, kernel=kernel, gamma=gamma)
# model = KerasRegressor(build_fn=create_model, batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=3)
# grid = GridSearchCV(estimator=pipe, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)
result = grid.fit(X, Y)
print('Best SVM Hyper-parameters via grid search (combined_dataset):', grid.best_params_)


"""
from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],
}

# Initialize the SVC classifier
svc = SVC()

# Perform the grid search
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1')

# Fit the grid search to the training data
grid_search.fit(X, Y)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

"""
