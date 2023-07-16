import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df1=pd.read_csv('kidney_disease.csv')

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df1['rbc']=lb.fit_transform(df1['rbc'])
df1['pc']=lb.fit_transform(df1['pc'])
df1['pcc']=lb.fit_transform(df1['pcc'])
df1['ba']=lb.fit_transform(df1['ba'])
df1['htn']=lb.fit_transform(df1['htn'])
df1['dm']=lb.fit_transform(df1['dm'])
df1['cad']=lb.fit_transform(df1['cad'])
df1['appet']=lb.fit_transform(df1['appet'])
df1['pe']=lb.fit_transform(df1['pe'])
df1['ane']=lb.fit_transform(df1['ane'])
df1['classification']=lb.fit_transform(df1['classification'])


df1.replace('\t?', float('nan'), inplace=True)  # Replace '\t?' with NaN

# Convert the relevant columns to float
columns_to_convert = [ 'bp',     'sg',   'al' ,  'su',  'rbc',  'pc',  'pcc' , 'ba'  ,'bgr', 'bu', 'sc', 'sod', 'pot' ,'hemo' ,'pcv' , 'wc' , 'rc' ,'htn',  'dm'  ,'cad',  'appet' , 'pe' , 'ane' , 'classification']  # Replace with the actual column names
for column in columns_to_convert:
    df1[column] = pd.to_numeric(df1[column], errors='coerce')

# Drop rows with missing values
df1.dropna(inplace=True)



from fancyimpute import KNN

knn_imputer = KNN()
df1 = knn_imputer.fit_transform(df1)

df1=pd.DataFrame(df1,columns=['id',   'age',   'bp',     'sg',   'al' ,  'su',  'rbc',  'pc',  'pcc' , 'ba'  ,'bgr', 'bu', 'sc', 'sod', 'pot' ,'hemo' ,'pcv' , 'wc' , 'rc' ,'htn',  'dm'  ,'cad',  'appet' , 'pe' , 'ane' , 'classification'])


X1 = df1.drop('classification', axis=1)
y1 = df1['classification'][0:203]


from sklearn.model_selection import train_test_split
X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size=0.3,random_state=101)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X1_train,y1_train)


pickle.dump(rf,open('kidney1.pkl','wb'))