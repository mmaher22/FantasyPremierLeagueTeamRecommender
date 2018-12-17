
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import math


# In[2]:

#Read Input Dataset
data = pd.read_csv('mergedDataset.csv')
data = data[['id', 'player_position', 'player_name', 'last10_ratio_cleanSheets_opp', 'last10_ratio_cleanSheets_own', 'last10_ratio_wins_opp', 'last10_ratio_wins_own', 'last3_assists', 'last3_goals', 'last3_ratio_points', 'last3_ycards', 'opp_team_rank', 'player_team_rank', 'player_value', 'ratio_assists', 'ratio_attempted_passes', 'ratio_big_chancesCreated', 'ratio_big_chancesMiss', 'ratio_creativity', 'ratio_dribbles', 'ratio_fouls', 'ratio_goals_conceded_opp_team', 'ratio_goals_conceded_player_team', 'ratio_goals_opp_team', 'ratio_goals_player_team', 'ratio_goals_scored', 'ratio_key_passes', 'ratio_leading_goal', 'ratio_minutes_played', 'ratio_offsides', 'ratio_open_playcross', 'ratio_own_goals', 'ratio_penalties_conceded', 'ratio_penalties_missed', 'ratio_penalties_saved', 'ratio_saves', 'ratio_selection', 'ratio_tackles', 'ratio_threat', 'week_no', 'season', 'week_points']]


# In[3]:

#Split Training & Testing Set
data_refined = data.loc[:,[i for i in list(data.columns) if i not in ['id', 'player_name', 'player_team', 'season']]]
#one-hot encoding for player position
data_refined = pd.get_dummies(data_refined, columns = ['player_position', 'week_no'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(data_refined.loc[:, data_refined.columns != 'week_points'], data_refined['week_points'], test_size=0.2, random_state=0)


# In[4]:

#dict to store results:
results_train = {}
results_test = {}


# In[ 5]:

#Random Forest
reg = RandomForestRegressor(random_state=0, criterion = 'mse')
#Apply grid search for best parameters
params = {'randomforestregressor__n_estimators' : range(100, 500, 200),
          'randomforestregressor__min_samples_split' : range(2, 10, 3)}
pipe = make_pipeline(reg)
grid = GridSearchCV(pipe, param_grid = params, scoring='mean_squared_error', n_jobs=-1, iid=False, cv=5)
reg = grid.fit(X_train, y_train)
print('Best MSE: ', grid.best_score_)
print('Best Parameters: ', grid.best_estimator_)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)
results_train['random_forest'] = tr_err
results_test['random_forest'] = ts_err


# In[6]:

reg = GradientBoostingRegressor(random_state=0, n_estimators = 100, min_samples_split = 5).fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)


# In[7]:

#Lasso Linear Regression
reg = linear_model.Lasso()
#Apply grid search for best parameters
params = {'lasso__alpha':[0.001, 0.01, 0.1, 0.5, 0.9]}
pipe = make_pipeline(reg)
grid = GridSearchCV(pipe, param_grid = params, scoring='mean_squared_error', n_jobs=-1, iid=False, cv=5)
reg = grid.fit(X_train, y_train)
print('Best MSE: ', grid.best_score_)
print('Best Parameters: ', grid.best_estimator_)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)
results_train['Lasso_regression'] = tr_err
results_test['Lasso_regression'] = ts_err


# In[8 ]:

#Ridge Linear Regression
reg = Ridge()
#Apply grid search for best parameters
params = {'ridge__alpha':[0.0001, 0.01, 0.1, 0.5, 0.9],
          'ridge__solver':['svd', 'cholesky', 'lsqr', 'sag']}
pipe = make_pipeline(reg)
grid = GridSearchCV(pipe, param_grid = params, scoring='mean_squared_error', n_jobs=-1, iid=False, cv=5)
reg = grid.fit(X_train, y_train)
print('Best MSE: ', grid.best_score_)
print('Best Parameters: ', grid.best_estimator_)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)
results_train['Ridge_regression'] = tr_err
results_test['Ridge_regression'] = ts_err


# In[9 ]:

#SVM Regression
reg = SVR()
#Apply grid search for best parameters
params = {'svr__kernel':['linear', 'rbf'],
          'svr__C':[1, 10, 100],
          'svr__gamma':[1, 0.1, 0.01]}
pipe = make_pipeline(reg)
grid = GridSearchCV(pipe, param_grid = params, scoring='mean_squared_error', n_jobs=-1, iid=False, cv=5)
reg = grid.fit(X_train, y_train)
print('Best MSE: ', grid.best_score_)
print('Best Parameters: ', grid.best_estimator_)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)
results_train['SVM'] = tr_err
results_test['SVM'] = ts_err


# In[10]:

#Linear Regression
reg = LinearRegression()
#Apply grid search for best parameters
params = {}
pipe = make_pipeline(reg)
grid = GridSearchCV(pipe, param_grid = params, scoring='mean_squared_error', n_jobs=-1, iid=False, cv=5)
reg = grid.fit(X_train, y_train)
print('Best MSE: ', grid.best_score_)
print('Best Parameters: ', grid.best_estimator_)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)
results_train['Linear_regression'] = tr_err
results_test['Linear_regression'] = ts_err


# In[11 ]:

#Decision Tree Regressor
reg = DecisionTreeRegressor().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)


# In[ 12]:

#KNN
reg = KNeighborsRegressor()
params = {'kneighborsregressor__n_neighbors':[1, 5, 10, 20, 25]}
pipe = make_pipeline(reg)
grid = GridSearchCV(pipe, param_grid = params, scoring='mean_squared_error', n_jobs=-1, iid=False, cv=5)
reg = grid.fit(X_train, y_train)
print('Best MSE: ', grid.best_score_)
print('Best Parameters: ', grid.best_estimator_)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)


# In[ 13]:

#GBR
reg = GradientBoostingRegressor()
params = {'gradientboostingregressor__n_estimators' : range(100, 500, 200),
          'gradientboostingregressor__min_samples_split' : range(2, 10, 3)}
pipe = make_pipeline(reg)
grid = GridSearchCV(pipe, param_grid = params, scoring='mean_squared_error', n_jobs=-1, iid=False, cv=5)
reg = grid.fit(X_train, y_train)
print('Best MSE: ', grid.best_score_)
print('Best Parameters: ', grid.best_estimator_)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
tr_err = mean_squared_error(y_train_pred, y_train)
ts_err = mean_squared_error(y_test_pred, y_test)
print(tr_err, ts_err)

#end

