import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


data= pd.read_csv('heart.csv')


data=pd.get_dummies(data)


X=data.drop('target',axis=1)
y=data['target']

from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(X,y,test_size=0.3,random_state=101)


from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier()
dt.fit(X_train,y_train)

pickle.dump(dt,open('heart1.pkl','wb'))




