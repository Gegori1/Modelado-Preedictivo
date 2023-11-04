<h1 style="text-align: center;">Neural Networks</h1>

<!-- \title{Neural Networks}
\maketitle -->


<p style="text-align: center;">Predictive modeling</p>

<!-- \begin{center}
Predictive modeling
\end{center} -->

![](https://upload.wikimedia.org/wikipedia/commons/2/2d/Logo-ITESO-Vertical-SinFondo-png.png)

<p style="text-align: center;">Gregorio Alvarez</p>

<!-- \begin{center}
Gregorio Alvarez
\end{center} -->

<div style="page-break-after: always"></div>

<!-- \newpage -->

## Introduction

This report aims to evaluate the predictive power, benefits, and drawbacks of using a neural network model in a regression problem that requires a complex solution. The primary objective is to analyze the input variables statistically, test multiple linear models as benchmarks, compare their performance with a non-optimized neural network model, and ultimately utilize a grid search to identify the optimal hyperparameters for the optimizer function. The technical procedures involved in finding a regression model that can effectively fit the "Appliances Energy Prediction" dataset will be outlined.

## Dataset description

The data used for this analysis is the [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) dataset from the UC Irvine Machine Learning Repository. The dataset contains 19,735 observations and 29 variables, including the target variable. The target variable is a continuous variable representing the energy consumption in Wh of appliances in a low energy building. The remaining variables are a mix of continuous and categorical variables, and are described in the table below.

| Variable    | Description                                                      | Type        |
|-------------|------------------------------------------------------------------|-------------|
| date        | date in format "yyyy-mm-dd hh:mm:ss"                             | categorical |
| Appliances  | energy consumption in Wh of appliances                           | continuous  |
| lights      | energy consumption in Wh of light fixtures                       | continuous  |
| T1          | temperature in kitchen area in Celsius                           | continuous  |
| RH_1        | humidity in kitchen area, in percentage                          | continuous  |
| T2          | temperature in living room area in Celsius                       | continuous  |
| RH_2        | humidity in living room area, in percentage                      | continuous  |
| T3          | temperature in laundry room area in Celsius                      | continuous  |
| RH_3        | humidity in laundry room area, in percentage                     | continuous  |
| T4          | temperature in office room in Celsius                            | continuous  |
| RH_4        | humidity in office room, in percentage                           | continuous  |
| T5          | temperature in bathroom in Celsius                               | continuous  |
| RH_5        | humidity in bathroom, in percentage                              | continuous  |
| T6          | temperature outside the building (north side) in Celsius         | continuous  |
| RH_6        | humidity outside the building (north side), in percentage        | continuous  |
| T7          | temperature in ironing room in Celsius                           | continuous  |
| RH_7        | humidity in ironing room, in percentage                          | continuous  |
| T8          | temperature in teenager room 2 in Celsius                        | continuous  |
| RH_8        | humidity in teenager room 2, in percentage                       | continuous  |
| T9          | temperature in parents room in Celsius                           | continuous  |
| RH_9        | humidity in parents room, in percentage                          | continuous  |
| T_out       | temperature outside (from Chievres weather station) in Celsius   | continuous  |
| Press_mm_hg | pressure (from Chievres weather station), in mm Hg               | continuous  |
| RH_out      | humidity outside (from Chievres weather station), in percentage  | continuous  |
| Windspeed   | wind speed (from Chievres weather station), in m/s               | continuous  |
| Visibility  | visibility (from Chievres weather station), in km                | continuous  |
| Tdewpoint   | dew point temperature (from Chievres weather station) in Celsius | continuous  |
| rv1         | random variable 1, unrelated to other variables                  | continuous  |
| rv2         | random variable 2, unrelated to other variables                  | continuous  |

For this report, the date and random variables were removed from the dataset. The target variable was constructed from the addition of the Appliances and lights variables. The remaining variables were used as input variables for the models.

<div style="page-break-after: always"></div>

## Methods

### Analysis

A pairplot was generated to examine the distribution of variables and their relationships with each other. The pairplot revealed that the majority of the variables had a unimodal distribution, several independent variables exhibited high correlation, while the dependent variable was highly skewed and showed little to no correlation with the input variable.

![Pairplot](pairplot.png)

Figure 1: Pairplot of all variables


To gain a more accurate understanding of the degree of correlation between the variables, an absolute correlation plot was also obtained. This plot further confirmed the information obtained from the pairplot.

![Correlation plot](correlation.png)

Figure 2: Correlation plot
Gegori1/DL_Specialization
### Benchmark models

Seven linear models were trained to check their predictive power.

- Linear regression, with and without standardization:

A linear regression model was fitted with the predictors and no transformation. It was found that the model was slower to train, than the one with standardization transformation, but the predictive power remained unchanged. Therefore, this transformation was applied to the rest of the models. 

![Linear model. Real vs Predicted](real_predicted_lr.png)

Figure 3: Linear model. It can be observed that the large skewness present in the output variable leads to large errors for large values of the variable. 

- Linear regression, with variable selection:

A variable selection by highest correlation between pairs, with a threshold of 0.7, was applied to the data. Which decreased the number of features by a factor of 3. The selected variables were: `RH_2, RH_5, T8, RH_9, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint`. As can be seen in the correlation_plot, the majority of these variables, had low linear correlation with respect to the dependent variable, indicating a possible drop of predictive power from the remaining variables.

- Partial Least Squares (PLS) Regression:

The data was fitted using PLS regression. An iterative process was employed to determine the optimal number of components. It was determined that 12 components provided the best results, indicating the "sweet spot" for this analysis.

![PLS score vs number of components](pls_score_components.png)

Figure 4. Score vs number of components for the PLS regression

- Transformation to target variable:

Due to high skewness of the target variable, a logarithmic and a square transformation to this variable was applied following the next procedure.
The output variables was transformed after partition, the linear model was applied and the resulting prediction was transformed inversely. The obtained variable was measured against the real values.

- Lasso Regression:

A lasso regression, with default $\alpha$ parameter, was used to fit the data.


- Results

| Model | Test R2 | Train R2 | Test RMSE | Train RMSE |
|-------|------------|-------------|-----------|------------|
| Regression original data | 0.1544 | 0.1537 | 93.8604 | 96.5442 |
| Regression normalized data | 0.1544 | 0.1537 | 93.8604 | 96.5442 |
| Regression with selected features | 0.0325 | 0.0298 | 100.3980 | 103.3671 |
| PLS regression | 0.1546 | 0.1521 | 93.8516 | 96.6372 |
| Regression log transformed target | 0.0790 | 0.0968 | 97.9600 | 99.7386 |
| Regression root square transformed target | 0.1427 | 0.1383 | 94.5070 | 97.4197 |
| Lasso regression | 0.1533 | 0.1523 | 93.9259 | 96.6244 |

As can be seen from the previous table, the regression with the original data, the normalized data and the PLS regression have the highest accuracy, being the PLS regression the one with the highest interpretability thanks to the reduce number of components and the possibily of interpreting the coefficients of the components.

### Neural Network models:

A neural network network with 400 hidden neurons, hyperbolic tangent activation function, Adam optimizer and mean squared error as loss function was trained. The model was trained for 400 epochs, with  a learning rate of 0.01. To avoid saturation of the activation function, the data was normalized using the standardization method. To avoid overfitting, an early stop condition was set when the validation metric did not improve in 10 epochs. The following table shows the architecture of the neural network.

![Table with number of parameters and architecture](nn_architecture.png)

Figure 5: Table with number of parameters and architecture

The following table shows the results obtained from the neural network model:

| Model | Test R2 | Train R2 | Test RMSE | Train RMSE |
|-------|------------|-------------|-----------|------------|
| Neural network | 0.1544 | 0.1537 | 93.8604 | 96.5442 |

Since the train and test scores are further apart a second experiment with a train, test and validation set was performed. The results are shown in the following table:

| Model | Test R2 | Train R2 | Validation score | Test RMSE | Train RMSE | Validation RMSE |
|-------|------------|-------------|------------------|-----------|------------|-----------------|
| Neural network | 0.1544 | 0.1537 | 0.1544 | 93.8604 | 96.5442 | 93.8604 |

- Grid search

A grid search, with a ten fold, was performed to find the optimal hyperparameters for the neural network. The following table shows the results obtained from the grid search:


| Model | Test R2 | Train R2 | Test RMSE | Train RMSE | Best parameters |
|-------|------------|-------------|-----------|------------|-----------------|

### Discussion

All the neural network models outperformed the linear models by a large margin. It was found that the neural network needs high variability in the model in order to improve on the test and evaluation set and could be related to the high complexity of the problem. The grid search process indicate that the optimal hyperparameters were used previous to this process.



## Conclusions

It was proven that the Neural Networks can handle the high dimensionality of the data, and highly skewed output. In future studies, it would be interesting to study the output variable as a classification-regresssion problem, to handle the high skewness of the output variable. It would also be interesting to study the effect of the variable selection with PLS and logistic regression applied to these variables, as it is expected that the predictive power of the model would increase, due to the reduction of the dimensionality of the problem.




<!-- No variables were removed from the dataset, coming from the assumption, that the NN would chose the right variables 
 so to not lose information, that could be lose due to variable selection, letting the ne

By applying practically no data pre-processing to the data inputted to the neural network and the imporve in performance, it can be deducted the power of 

For future studies data preprocessing will be applied to compare the performance and the resources needed to accomplish the same task, as it is suspected that the high dimensionality of the neural network is diminishing the effects of the poorly pre-processed data. -->

