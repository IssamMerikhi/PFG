"""
BOX GOAL : IMPORT ALL DEPENDENCIES.
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from IPython.display import display
from matplotlib import *
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
import plotly.io as pio



"""
BOX GOAL : IMPORT THE DATA AND DISPLAY A COUPLE OF ROWS
"""

# Read data and drop redundant column.
data = pd.read_csv('PL.csv')

# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = data.drop(['FTR'],1)
y_all = data['FTR']

# Standardising the data.
from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
cols = [['H_ST','H_SOG','H_SFG','H_PT','H_COR','H_FL','H_YC','H_RC','A_ST','A_SOG','A_SFG','A_PT','A_COR','A_FL','A_YC','A_RC']]
for col in cols:
    X_all[col] = scale(X_all[col])

X_all = data.drop(['HTAG','HTHG','FTAG','FTHG','HTR','Date','FTR','Country','League','Type','Season','Home_Team','Away_team','ETR','ETHG','ETAG','PENR','PENHG','PENAG'],1)

from sklearn.feature_selection import SelectKBest, chi2

sel = SelectKBest(chi2, k = 5)
sel.fit(X_all, y_all)
X = sel.transform(X_all)
X = pd.DataFrame(X)


from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y_all, 
                                                    test_size = 0.20,
                                                    random_state = 3,
                                                    stratify = y_all)

from sklearn.model_selection import GridSearchCV

logModel = LogisticRegression()
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

LogGrid = GridSearchCV(logModel, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)

BestLog = LogGrid.fit(X_train,y_train)


from sklearn.model_selection import GridSearchCV


max_features_range = np.arange(1,6,1)
n_estimators_range = np.arange(10,210,10)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
rf = RandomForestClassifier()
RandGrid = GridSearchCV(estimator=rf, param_grid=param_grid, cv = 5)

RandGrid.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

SuperVector = SVC()
param_grid = {'C':[0.1,1], 'kernel':['rbf', 'poly', 'sigmoid','linear'], 'degree':[1,2,3]}
SVMGrid=GridSearchCV(estimator=SuperVector, param_grid=param_grid)

SVMGrid.fit(X_train, y_train)

LogPred = LogGrid.predict(X_test)
RandPred = RandGrid.predict(X_test)
SVMPred = SVMGrid.predict(X_test)
res = pd.DataFrame([LogPred, RandPred, SVMPred, y_test]).transpose().head(10) #this is my prediction (0:A_pred, etc.)
res.columns = ['LogPred', 'RandPred', 'SVMPred', 'Y']

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from IPython.display import HTML

Y_train = to_categorical(y_train, 3)
Y_test = to_categorical(y_test, 3)

count_classes = Y_test.shape[1]

model = Sequential()
model.add(Dense(1000, activation='relu', input_dim=5))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=20)



pred_train= model.predict(X_train)
scores = model.evaluate(X_train, Y_train, verbose=0)
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, Y_test, verbose=0)