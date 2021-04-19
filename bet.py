import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from matplotlib import *
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm

data = pd.read_csv('PL.csv')

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

SuperVector = svm.SVC()
param_grid = {'C':[0.1,1], 'kernel':['rbf', 'poly', 'sigmoid','linear'], 'degree':[1,2,3]}
SVMGrid=GridSearchCV(estimator=SuperVector, param_grid=param_grid)
SVMGrid.fit(X_train, y_train)
SVMPred = SVMGrid.predict(X_test)


Man_City = Image.open('logo/man_city_logo.png')
Arsenal = Image.open('logo/arsenal_logo2.png')

st.set_page_config(page_title='PibeOro')


#col1, col2 = st.beta_columns([3,3])

teams = {'Man_City': Man_City, 
         'Arsenal': Arsenal}
first_team = st.selectbox('1st team', list(teams.keys()))
st.image(teams[first_team], width=200)
second_team = st.selectbox('2nd team', list(teams.keys()))
st.image(teams[second_team], width=200)














"""
man_city = pd.DataFrame([3,4,2,1,8]).transpose()
arsenal = pd.DataFrame([0,1,2,4,4]).transpose()

if (SVMGrid.predict(man_city) > SVMGrid.predict(arsenal)):
    st.write('MANCHESTER CITY IS THE WINNER')
    st.image(man_city_logo, width=200)

if (SVMGrid.predict(man_city) < SVMGrid.predict(arsenal)):
    st.write('ARSENAL IS THE WINNER')
    st.image(arsenal_logo, width=400)

if (SVMGrid.predict(man_city) == SVMGrid.predict(arsenal)):
    st.write('DRAW')

"""