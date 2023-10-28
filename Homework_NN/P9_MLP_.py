# -*- coding: utf-8 -*-

#pip install keras
#pip install tensorflow

#pip install pydot
#pip install graphviz

# Libraries
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import plot_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score,mean_squared_error,make_scorer
from sklearn.model_selection import train_test_split

#%% Import dataset
data = pd.read_csv('../Data/Airfoil_Self_Noise/airfoil_self_noise.dat',
                   sep = '\t',
                   header = None)
names = ['Freq','Angle','Chord','Velocity','Suction','Sound']
data.columns=names

# Data input and output selection
X = data[names[0:5]]
Y = data[names[5]]

# Scaling dataset
# from sklearn import preprocessing
# X = preprocessing.scale(X)
# Y = preprocessing.scale(Y)

# from sklearn.preprocessing import MinMaxScaler
# X = MinMaxScaler().fit_transform(X)
# Y = MinMaxScaler().fit_transform(np.resize(Y,(1503,1)))

#%% Dataset split (Training and Test)

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)

#%% Model Design (Neural Network)
epochs = 400
learning_rate = 0.01
decay_rate = learning_rate/epochs
momentum = 0.8

# Neural network architecture
model = Sequential([tf.keras.Input(shape = (5,),name='input_layer'),
                    Dense(10, activation='tanh'),
                    Dense(1, activation='linear')
                    ])

# Optimizer configuration
opt = keras.optimizers.SGD(learning_rate=learning_rate)
                            # momentum=momentum,
                           # decay=decay_rate,
                           # nesterov=True)
# opt = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss = 'mean_squared_error',
              optimizer=opt,
              metrics=['mse'])

#%% Fit the model
model_history = model.fit(xtrain,ytrain,
                    epochs=epochs,
                    batch_size=200,
                    validation_data=(xtest,ytest))

#%% Performance evaluation
score = model.evaluate(xtest,ytest)
print('Test loss:', score[0])
print('Test mse:', score[1])

#%% Plot the loss function and the metrics
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(model_history.history['loss'], 'r', label='train')
ax.plot(model_history.history['val_loss'], 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

# Plot the mse
fig, ax = plt.subplots(1, 1, figsize=(10,6))
# ax.plot(np.sqrt(model_history.history['mean_squared_error']), 'r', label='train')
# ax.plot(np.sqrt(model_history.history['val_mean_squared_error']), 'b' ,label='val')
ax.plot(np.sqrt(model_history.history['mse']), 'r', label='train')
ax.plot(np.sqrt(model_history.history['val_mse']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'MSE', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)


#%%
yhat = model.predict(xtrain)

R2_score = r2_score(ytrain,yhat)

xmin,xmax = min(ytrain),max(ytrain)
xline = np.linspace(xmin,xmax)
fig = plt.figure(figsize=(10,6))
plt.scatter(ytrain,yhat,label='Estimation')
plt.plot(xline,xline,'k--',label='Perfect estimation')
plt.xlabel('Real output', fontsize=20)
plt.ylabel('Estimation output', fontsize=20)
plt.title('R^2=%0.4f'%R2_score, fontsize=20)
plt.legend()
plt.grid()
plt.show()

#%% Neural network weights
W = model.layers[0].get_weights()

#%% View the model
plot_model(model, show_shapes=True)
# plot_model(model, to_file='../figures/P4_fig/model.png', show_shapes=True)

#%% Tune Hyperparameters
from sklearn.model_selection import GridSearchCV
# RandomSearchCV

def create_model(lr=0.1,momentum=0.8):
    # Neural network architecture
    model = Sequential()
    model.add(Dense(10,activation='tanh',input_dim=5))
    model.add(Dense(1,activation='linear'))
    
    # Optimizer configuration
    opt = keras.optimizers.SGD(learning_rate=lr,momentum=momentum,
                               nesterov=True)
    model.compile(loss = 'mean_squared_error',
                  optimizer=opt,
                  metrics=['mse'])
    return model

# Fixed hyperparameters
epochs = 100
learning_rate = 0.1
decay_rate = learning_rate/epochs
momentum = 0.8
batch_size = 200

model_search = KerasRegressor(build_fn=create_model,epochs=epochs)

# define the grid search parameters
lr = [0.1, 0.05, 0.01]
momentum = [0.8, 0.6, 0.4]
param_grid = dict(lr=lr,momentum=momentum)

selection_score = make_scorer(r2_score)
# selection_score = make_scorer(mean_squared_error,greater_is_better=False)

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid = GridSearchCV(estimator=model_search, param_grid=param_grid,
                    cv=2,return_train_score=True,
                    scoring=selection_score)
grid_result = grid.fit(xtrain, ytrain)

#%% print results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f' mean={mean:.4}, std={stdev:.4} using {param}')




#%% Save a model to JSON

# serialize model to JSON
model_json = model.to_json()

with open("../figures/P4_fig/model.json", "w") as json_file:
    json_file.write(model_json)

# save weights to HDF5
model.save_weights("../figures/P4_fig/model.h5")
print("Model saved")

#%% Retrieve the model: load json and create model
from keras.models import model_from_json
json_file = open('../figures/P4_fig/model.json', 'r')
saved_model = json_file.read()
# close the file as good practice
json_file.close()
model_ = model_from_json(saved_model)
# load weights into new model
model_.load_weights("../figures/P4_fig/model.h5")
print("Model loaded")
























