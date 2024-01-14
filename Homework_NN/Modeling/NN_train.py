# %%
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import pendulum as pen

from neural_networks import Model
from neural_networks.nn_utils import SaveBestModelCallback
from neural_networks.Utils import read_config, find_root_dir
# %%

cfg = read_config()

rd = find_root_dir()

data_path = rd / cfg['data']['path'] / cfg['data']['train_file']

# %%

model_name = (
    f"{cfg['model']['name']}"
    '_'
    f"{cfg['model']['type']}"
    '_'
    f"{cfg['model']['version']}"
)

model_path = cfg['model']['save_name'] + pen.now().to_datetime_string()
random_state = cfg['model']['random_state']
train_test_valid_split = cfg['model']['split']

data = pd.data = (
    pd.read_csv(data_path)
    .drop(['date', 'rv1', 'rv2'], axis=1)
    .assign(target = lambda k: k.lights + k.Appliances)
    .drop(['lights', 'Appliances'], axis=1)
)

# %% Traint, test and validation split
X_train, X_rest, y_train, y_rest = train_test_split(
    data.drop('target', axis=1),
    data.target,
    test_size=1 - train_test_valid_split[0],
    random_state=random_state
)

X_test, X_val, y_test, y_val = train_test_split(
    X_rest,
    y_rest,
    test_size=(
        train_test_valid_split[2] 
        / (train_test_valid_split[1] + train_test_valid_split[2])
    ),
    random_state=random_state
)
# -

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# %%
def create_model(model_name):
    if model_name == 'neural_network_regressor_0.0.1':
        return Model.NNRegressor(activation="tanh", input_shape=500)
    else:
        raise NotImplementedError("Model not implemented yet.")


model = create_model(model_name)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3),
    loss='mse'
)
# -

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=10
)

mc = SaveBestModelCallback(
    threshold=4700,
    filepath=model_path,
    monitor_metric='val_loss',
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    use_multiprocessing=True,
    callbacks = [es, mc]
)


# %% Plot training and validation loss
plt.plot(history.history['loss'][-100:])
plt.plot(history.history['val_loss'][-100:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# %%
# compute r2 score
from sklearn.metrics import r2_score
# train
y_pred_train = model.predict(X_train)
# validation
y_pred_val = model.predict(X_val)
# test
y_pred_test = model.predict(X_test)

r2_score(y_train, y_pred_train), r2_score(y_val, y_pred_val), r2_score(y_test, y_pred_test)

# %%
# plot real vs predicted
plt.scatter(y_test, y_pred_test)
max_size = max(y_test.max(), y_pred_test.max())
plt.plot([0, max_size], [0, max_size], 'k--')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

# plot real vs predicted
plt.scatter(y_train, y_pred_train)
max_size = max(y_train.max(), y_pred_train.max())
plt.plot([0, max_size], [0, max_size], 'k--')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

# plot real vs predicted
plt.scatter(y_val, y_pred_val)
max_size = max(y_val.max(), y_pred_val.max())
plt.plot([0, max_size], [0, max_size], 'k--')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()
# -



# %%
