# %% [markdown]
# ### Load Libraries

# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

# %%
[i for i in os.listdir() if i.endswith('.data')]

# %% [markdown]
# ### Load data

# %%
data = pd.read_csv('abalone.data', header=None)
data.columns = ["Sex", "Length", "Diam", "Height", "Whole", "Shucked", "Viscera", "Shell", "Rings"]
data.drop("Sex", axis=1, inplace=True)
data.head()

# %%
data.info()

# %%
data.describe().loc[["mean", "std", "min", "max"], :].style.background_gradient(axis=1)

# %%
sns.pairplot(data);

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# De la gráfico de pares se puede ver que la variable de altura tiene valores atipicos, por lo que se procedera a removerlos

# %%
from scipy.stats import zscore

# %%
index_drop = (
    data
    .assign(
        z_score = lambda k: k[["Height"]].apply(zscore).abs()
    )
    .query("z_score > 3")
    .index
)

# %%
data = data.query("index not in @index_drop")

# %% [markdown]
# ### Fit model with all variables

# %%
data_train, data_test = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# %%
# scaler = StandardScaler()
# scaler.set_output(transform='pandas')

# %%
X_train, y_train = data_train.drop("Rings", axis=1), data_train["Rings"]
X_test, y_test = data_test.drop("Rings", axis=1), data_test["Rings"]

# %%
model = LinearRegression()
model.fit(X_train, y_train)

# %%
predict = model.predict(X_test)

# %%
# calculate r_squared and rmse
r_squared = model.score(X_test, y_test)
rmse = np.sqrt(np.mean((predict - y_test) ** 2))

print(f"R-squared test: {r_squared}")
print(f"RMSE test: {rmse}")

# %%
predict_train = model.predict(X_train)

# %%
# calculate r_squared and rmse for train
r_squared = model.score(X_train, y_train)
rmse = np.sqrt(np.mean((predict_train - y_train) ** 2))

print(f"R-squared train: {r_squared}")
print(f"RMSE train: {rmse}")

# %% [markdown]
# ### Fit model after removing correlated features

# %% [markdown]
# #### Check Correlation

# %%
data.corr().style.background_gradient(cmap='coolwarm')

# %%
correlation = X_train.corr()
sns.clustermap(correlation, method="complete", cmap='RdBu', annot=True, 
               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12));

# %% [markdown]
# Se puden obserar cuatro grupos de variables correlacionadas con un umbral de 0.5

# %% [markdown]
# #### Seleccionar variables

# %%
def selection_by_corr(dataset, threshold):
    corr_ = (dataset.corr() * -(np.identity(dataset.shape[1]) - 1)).abs()
    while corr_.max().max() > threshold:
        args = np.unravel_index(corr_.to_numpy().argmax(), corr_.shape)
        if corr_.iloc[args[0], :].mean() > corr_.iloc[:, args[1]].mean():
            name_drop = corr_.iloc[args[0], :].name
            corr_.drop(name_drop, axis=1, inplace=True)
            corr_.drop(name_drop, axis=0, inplace=True)
        else:
            name_drop = corr_.iloc[:, args[1]].name
            corr_.drop(name_drop, axis=1, inplace=True)
            corr_.drop(name_drop, axis=0, inplace=True)
    return corr_.columns.values

# %%
select_col = selection_by_corr(X_train, 0.9)
select_col

# %% [markdown]
# ### Fit model with selected columns

# %%
X_train_select = X_train[select_col]
X_test_select = X_test[select_col]

# %%
model = LinearRegression()
model.fit(X_train_select, y_train)

# %%
predict = model.predict(X_test_select)

# %%
# calculate r_squared and rmse
r_squared = model.score(X_test_select, y_test)
rmse = np.sqrt(np.mean((predict - y_test) ** 2))

print(f"R-squared test: {r_squared}")
print(f"RMSE test: {rmse}")

# %%
predict_train = model.predict(X_train_select)

# %%
# calculate r_squared and rmse for train
r_squared = model.score(X_train_select, y_train)
rmse = np.sqrt(np.mean((predict_train - y_train) ** 2))

print(f"R-squared train: {r_squared}")
print(f"RMSE train: {rmse}")

# %% [markdown]
# ### PCA

# %% [markdown]
# - Se crearan nuevas variables a partir de las variables originales, estas nuevas variables seran las variables originales.
# - Se seleccionaran variables en base al grado de varianza.

# %% [markdown]
# #### Compute PCA

# %%
# plot the variance explained for each principal component
pca = PCA().fit(X_train)

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# En base a la gráfica anterior, se seleccionan 3 componente principal

# %% [markdown]
# #### Seleccion de variables

# %%
pca = PCA(n_components=3)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# %% [markdown]
# #### Graficar

# %%
# Graficar las primeras 3 componentes principales
# contra la variable objetivo
fig = plt.subplots(2, 2)
plt.subplot(2, 2, 1)
plt.scatter(X_train_pca[:, 0], y_train)
plt.xlabel('PC1')
plt.ylabel('Rings')
plt.subplot(2, 2, 2)
plt.scatter(X_train_pca[:, 1], y_train)
plt.xlabel('PC2')
plt.ylabel('Rings')
plt.subplot(2, 2, 3)
plt.scatter(X_train_pca[:, 2], y_train)
plt.xlabel('PC3')
plt.ylabel('Rings')

# %% [markdown]
# #### Fit a model using the principal components

# %%
pca = PCA(n_components=3)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# %%
model = LinearRegression()
model.fit(X_train_pca, y_train)

# %%
predict = model.predict(X_test_pca)

# %%
# calculate r_squared and rmse
r_squared = model.score(X_test_pca, y_test)
rmse = np.sqrt(np.mean((predict - y_test) ** 2))

print(f"R-squared test: {r_squared}")
print(f"RMSE test: {rmse}")

# %%
# calculate r_squared and rmse for train
predict_train = model.predict(X_train_pca)
r_squared = model.score(X_train_pca, y_train)
rmse = np.sqrt(np.mean((predict_train - y_train) ** 2))

print(f"R-squared train: {r_squared}")
print(f"RMSE train: {rmse}")

# %% [markdown]
# ### PLS

# %% [markdown]
# - Se crearan nuevas variables independientes y dependiente en base a la transformación de las variables originales.
# - Se seleccionan las variables independiente hasta que las metricas decrementen.

# %%
pls = PLSRegression()
pls.fit(X_train, y_train)

predict = pls.predict(X_test)

# calculate r_squared and rmse
r_squared = pls.score(X_test, y_test)
rmse = np.sqrt(np.mean((predict.reshape(-1) - y_test) ** 2))

print(f"R-squared test: {r_squared}")
print(f"RMSE test: {rmse}")

# %% [markdown]
# #### Plot

# %%
pls = PLSRegression(n_components=4)
pls.fit(X_train, y_train)

X_train_pls = pls.transform(X_train)
X_test_pls = pls.transform(X_test)

# %%
# Graficar las primeras 3 componentes principales
# contra la variable objetivo
fig = plt.subplots(2, 2)
plt.subplot(2, 2, 1)
plt.scatter(X_train_pls[:, 0], y_train)
plt.xlabel('PLS1')
plt.ylabel('Rings')
plt.subplot(2, 2, 2)
plt.scatter(X_train_pls[:, 1], y_train)
plt.xlabel('PLS2')
plt.ylabel('Rings')
plt.subplot(2, 2, 3)
plt.scatter(X_train_pls[:, 2], y_train)
plt.xlabel('PLS3')
plt.ylabel('Rings')
plt.subplot(2, 2, 4)
plt.scatter(X_train_pls[:, 3], y_train)
plt.xlabel('PLS4')
plt.ylabel('Rings')

# %%
# Remove variables iteratively
for i in range(X_train.shape[1], 0, -1):
    pls = PLSRegression(n_components=i)
    pls.fit(X_train, y_train)
    print(f"Number of components: {i}")
    # predict train
    predict_train = pls.predict(X_train)
    r_squared = pls.score(X_train, y_train)
    rmse = np.sqrt(np.mean((predict_train.reshape(-1) - y_train) ** 2))
    print(f"R-squared train: {r_squared:.4f}")
    print(f"RMSE train: {rmse:.4f}")
    # predict test
    predict = pls.predict(X_test)
    r_squared = pls.score(X_test, y_test)
    rmse = np.sqrt(np.mean((predict.reshape(-1) - y_test) ** 2))
    print(f"R-squared test: {r_squared:.4f}")
    print(f"RMSE test: {rmse:.4f}")

# %% [markdown]
# Existe un cambio significativo ahasta las 2 componentes, por lo que se utilizara un modelo con 3 componentes.

# %%
pls = PLSRegression(n_components=3)
pls.fit(X_train, y_train)

# %%
predict_train = pls.predict(X_train)
r_squared = pls.score(X_train, y_train)
rmse = np.sqrt(np.mean((predict_train.reshape(-1) - y_train) ** 2))
print(f"R-squared train: {r_squared:.4f}")
print(f"RMSE train: {rmse:.4f}")
# predict test
predict = pls.predict(X_test)
r_squared = pls.score(X_test, y_test)
rmse = np.sqrt(np.mean((predict.reshape(-1) - y_test) ** 2))
print(f"R-squared test: {r_squared:.4f}")
print(f"RMSE test: {rmse:.4f}")

# %%



