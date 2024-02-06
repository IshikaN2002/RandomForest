# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:18:27 2024

@author: hp
"""
'''
Build a Decision Tree & Random Forest model on the
 fraud data. Treat those who have taxable_income <= 30000
 as Risky and others as Good (discretize the taxable income column).
 '''
import pandas as pd
fraud = pd.read_csv("Fraud_check.csv")
dir(fraud)

df = pd.DataFrame(fraud)
df.head()

df['Undergrad'] =fraud.Undergrad
df[0:12]

X = df.drop('Undergrad',axis='columns')
y = df.Undergrad

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)

model.fit(X_train, y_train)

model.score(X_test, y_test)
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm

#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')