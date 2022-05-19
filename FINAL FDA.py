"""
Created on Wed April 20 18:22:15 2022
Final Assignment
@author: Novia Widya Chairani
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as optimization

#import machine learning packages
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

#############################################################################
# Portfolio's stock list
# MMM=3M, ABT=Abbott, ACN=Accenture, ARE=Alexandria Real Estate, AMZN=Amazon
stocks = ['MMM','ABT','ACN','ARE','AMZN']

# Set dates
start_date = dt.datetime(2017,2,1)
end_date = dt.datetime(2022,1,31) 

# Set lags
lags=2

# Get the stock data from Yahoo Finance
df = web.DataReader(stocks,data_source='yahoo',start=start_date,end=end_date)["Adj Close"]






#############################################################################
# QUESTION 1 
#############################################################################
##(a) Train your algorithm to predict an ‘up’ or ‘down’ day for your portfoliO
#############################################################################
# Calculating the Daily Portfolio Returns
returns = np.log(df/df.shift(1))
optimal_weights = [0.016,0.297,0.397,0.022,0.268]
daily_portfolio_returns = (returns*optimal_weights).sum(axis=1)

# Create a new dataframe
#we want to use additional features: lagged returns...today's returns, yesterday's returns etc
tslag = pd.DataFrame(index=daily_portfolio_returns.index)
tslag["Today"] = daily_portfolio_returns

# Create the shifted lag series of prior trading period close values
range(0, lags)
for i in range(0, lags):
    tslag["Lag%s" % str(i+1)] = daily_portfolio_returns.shift(i+1)

# Create the returns DataFrame
dfret = pd.DataFrame(index=tslag.index)
dfret["Today"] = tslag["Today"].pct_change()

# Create the lagged percentage returns columns
for i in range(0, lags):
    dfret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()
        
# Clearing up the Nan values
dfret.drop(dfret.index[:4], inplace=True)

# "Direction" column (+1 or -1) indicating an up/down day (0 indicates daily non mover)
dfret["Direction"] = np.sign(dfret["Today"])

# Replace where nonmover with down day (-1)
dfret["Direction"]=np.where(dfret["Direction"]==0, -1, dfret["Direction"] ) # up day and down day. 0 is an non mover and considered a down day then sell. buy back at a cheaper price

# Use the prior two days of returns as predictor 
# values, with todays return as a continuous response
x = dfret[["Lag1","Lag2"]]
y = dfret[["Today"]]

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Alternative test/train split
# The test data is split into two parts: Before and after 1st Jan 2018.
start_test = dt.datetime(2018,1,1)

# Create training and test sets
x_train = x[x.index < start_test]
x_test = x[x.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]

#####################################
#Regression
#####################################
# We use Decision Trees as the machine learning model
model=DecisionTreeRegressor(max_depth = 2, min_samples_leaf= 5)
# Train the model on the training set
results=model.fit(x_train, y_train)

plt.figure(figsize=(22,16))
plot_tree(results, filled=True)

# Make an array of predictions on the test set
y_pred = model.predict(x_test)
model.score(x_test, y_test)
# Predict an example
x_example=[[0,0.09]]
yhat=model.predict(x_example)

#####################################
#Classification
#####################################

# Plot log2 function (the measure of entropy)
plt.figure()
plt.plot(np.linspace(0.01,1),np.log2(np.linspace(0.01,1)))
plt.xlabel("P(x)")
plt.ylabel("log2(P(x))")
plt.show()






#############################################################################
## (b) Use 10 fold cross validation to prune your tree
#############################################################################
depth = []
for i in range(1,20): #i is same as the lambda.  test the depth
    model = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    # Perform 5-fold cross validation k=5
    scores = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
    depth.append((scores.mean(),i))
    
print(max(depth)) 






#############################################################################
## (c) Plot your decision tree 
#############################################################################

# Use the prior two days of returns as predictor 
# values, with direction as the discrete response
x = dfret[["Lag1","Lag2"]]
y = dfret["Direction"]

# Alternative test/train split
# The test data is split into two parts: Before and after 1st Jan 2018.
start_test = dt.datetime(2018,1,1)

# Create training and test sets
x_train = x[x.index < start_test]
x_test = x[x.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]

# We use Decision Trees as the machine learning model
model=DecisionTreeClassifier(criterion = 'entropy', max_depth = 2)
# Train the model on the training set
results=model.fit(x_train, y_train)

# Make an array of predictions on the test set
y_pred = model.predict(x_test)

# Predict an example
x_example=[[0,0.09]]
yhat=model.predict(x_example)

# Plot decision tree
dt_feature_names = list(x.columns)
dt_target_names = ['Sell','Buy'] #dt_target_names = [str(s) for s in y.unique()]
plt.figure(figsize=(22,16))
plot_tree(results, filled=True, feature_names=dt_feature_names, class_names=dt_target_names)
plt.show() 





#############################################################################
## (d) Acurracy of the model
#############################################################################
# Output the hit-rate and the confusion matrix for the model
print("Confusion matrix: \n%s" % confusion_matrix(y_pred, y_test))
print("Accuracy of decision tree model on test data: %0.3f" % model.score(x_test, y_test))






#############################################################################
# QUESTION 2
#############################################################################
## Ensemble methods

#####################################
# Bagging
#####################################
# Ensemble_model=BaggingClassifier(model, n_estimators=100, random_state=0)
ensemble_model=BaggingClassifier(model, n_estimators=100, random_state=0)

ensemble_model.fit(x_train, y_train)
y_pred = ensemble_model.predict(x_test)
yhat=ensemble_model.predict(x_example)
#print("Confusion matrix: \n%s" % confusion_matrix(y_pred, y_test))
print("Accuracy of bagging model on test data: %0.3f" % ensemble_model.score(x_test, y_test))

#####################################
# Random Forests
#####################################
# Ensemble_RF=RandomForestClassifier(n_estimators=100, random_state=0)
ensemble_RF=RandomForestClassifier(n_estimators=100, random_state=0) #max_features="sqrt"

ensemble_RF.fit(x_train, y_train)
y_pred = ensemble_RF.predict(x_test)
yhat=ensemble_RF.predict(x_example)
# Print("Confusion matrix: \n%s" % confusion_matrix(y_pred, y_test))
print("Accuracy of random forest model on test data: %0.3f" % ensemble_RF.score(x_test, y_test))


















































































