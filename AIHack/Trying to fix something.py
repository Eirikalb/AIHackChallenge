#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import model_selection,preprocessing
data_dir="/home/eirik/python/AIHack/data/California_metadata/"


# In[2]:

import pandas as pd
label_df=pd.read_csv(data_dir+"BG_METADATA_2016.csv")
columnnames={}
for row in label_df.iterrows():
    columnnames[row[1][1]]=row[1][2]


# In[3]:

#Main dataframe has all variables from this dataset
df=pd.read_csv(data_dir+"X15_EDUCATIONAL_ATTAINMENT.csv")
df.rename(columns=columnnames,inplace=True)
# This dataset contains the variable we want to predict
y_df=pd.read_csv(data_dir+"X19_INCOME.csv")
y_df.rename(columns=columnnames,inplace=True)

y_columnname="PER CAPITA INCOME IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS): Total: Total population -- (Estimate)"
#output_df.to_csv("Just for find an interresting variable.csv")
df[y_columnname]=y_df[y_columnname]
# Drop variables used for identification
df.drop("OBJECTID",axis=1,inplace=True)
df.drop("GEOID",axis=1,inplace=True)
# Handle na and split dataset again
df.dropna(inplace=True)
y=df[y_columnname]
df.drop(y_columnname,axis=1,inplace=True)
df.describe()

# In[4]:

# Normalize df values
columns=df.columns.values
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
for i in range(len(columns)):  
    df[columns[i]] = pd.DataFrame(x_scaled[i])


# In[5]:

import scipy
import matplotlib.pyplot as plt
import seaborn as sns

def cramers_corrected_stat(confusion_matrix):
    #Calculates the corrected Cramer's V statistic
    #Args: confusion_matrix: The confusion matrix of the variables to calculate the statistic on
    #Returns: The corrected Cramer'v V statistic
    chi2, _, _, _ = scipy.stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def numpy_minmax(X):
    xmin =  X.min(axis=0)
    return (X - xmin) / (X.max(axis=0) - xmin)

cols = list(df.columns.values)
corrM = np.zeros((1,len(cols)))
y_scaled=numpy_minmax(np.array(y))

# Calculate correlations of every variable against y
for col1 in cols:
    A, B = df[col1], y_scaled
    idx1 = cols.index(col1)
    dfObserved = pd.crosstab(A,B) 
    corrM[0,idx1] = cramers_corrected_stat(dfObserved.values)

corr = pd.DataFrame(corrM, index=cols, columns=cols)
# Mask to get lower triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.cubehelix_palette(light=1, as_cmap=True)

# Draw the heatmap with the mask 
fig = plt.figure(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()

# In[10]:
#Split dataset
X_train,X_test,y_train,y_test=model_selection.train_test_split(df,y,test_size=0.2,random_state=60)


# In[10]:

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_test, y_test)


# In[11]:

accuracies = model_selection.cross_val_score(estimator = reg, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())


# In[ ]:

import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(
    X_train,
    y_train,
    )
# In[ ]:

# In[ ]:
accuracies = model_selection.cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())

# In[ ]:
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
xgbreg=GradientBoostingRegressor()

parameters = [{"n_estimators": [1000], 'learning_rate':[0.2,0.1,0.05],"alpha":[0.95,0.9,0.8]}]
grid_search = GridSearchCV(estimator = xgbreg,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 4,
                           n_jobs = -1)
y_train=np.array(y_train)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_