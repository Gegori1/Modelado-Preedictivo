#%% Import libraries


import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


 
def scale_data(train: pd.DataFrame, test: pd.DataFrame, columns: list) -> (pd.DataFrame, pd.DataFrame):
    """
    Scale the data using the StandardScaler from sklearn
    """
    # create a scaler object
    scaler = StandardScaler()
    # fit the scaler to the training data
    if not isinstance(columns, list):
        columns = [columns]
    # transform both the train and the test data
    train_scaled = train.copy()
    test_scaled = test.copy()
    for col in columns:
        train_scaled[col] = scaler.fit_transform(train[[col]])
        test_scaled[col] = scaler.transform(test[[col]])
    return train_scaled, test_scaled

# %% Load data

data_train = pd.read_csv('data/Audit_train_.csv', index_col=0)
data_test = pd.read_csv('data/Audit_test_.csv', index_col=0)
data_unknown = pd.read_csv('data/Audit_unknown_.csv', index_col=0)

#%% Data transformation

# selected variables by correlation and accuracy of the resulting model
columns = ["PARA_B", "Inherent_Risk", "Audit_Risk"]


# points that were not found in the unknown data with classificator
filters = 'Audit_Risk < 60'


data_train_clean = (
    data_train
    [columns]
    .query(filters)
    .apply(np.log1p)
)
data_test_clean = (
    data_test
    [columns]
    .query(filters)
    .assign(
        PARA_B = lambda x: x['PARA_B'].apply(np.log1p),
        Inherent_Risk = lambda x: x['Inherent_Risk'].apply(np.log1p),
    )
)


data_train_clean, data_test_clean = scale_data(
    data_train_clean, data_test_clean, 
    [
        "PARA_B", 
        "Inherent_Risk"
    ]
)

#%% PLS



n_components = 2

# Fit model
model_pls = (
    PLSRegression(n_components=n_components)
    .fit(
        data_train_clean.drop("Audit_Risk", axis=1),
        data_train_clean['Audit_Risk']
    )
)

# Prediction
y_pred = model_pls.predict(data_test_clean.drop("Audit_Risk", axis=1)).flatten()

# Inverse transformation
y_pred = np.exp(y_pred) - 1


#%% Results visualization

# Residual plot
plt.plot(y_pred, y_pred - data_test_clean['Audit_Risk'], 'o')
plt.show()

# rmse
print(np.sqrt(np.mean((y_pred - data_test_clean['Audit_Risk'])**2)))
# r2
print(1 - np.sum((y_pred - data_test_clean['Audit_Risk'])**2) / np.sum((data_test_clean['Audit_Risk'] - data_test_clean['Audit_Risk'].mean())**2))

