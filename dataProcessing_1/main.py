import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

#loading the dataset 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

print("Printing the data from file")
print(X)
print(y)

#handling the 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
imputer.fit(X[: , 1:3])
X[: , 1:3] = imputer.transform(X[: , 1:3]);

print("Printing after handling the missing value")
print(X)

#encoding the categorial data 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder" , OneHotEncoder() , [0])] , remainder="passthrough")
X = np.array(ct.fit_transform(X));
print(X)

#encoding the dependent viablke vector
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y);
print("Printing after label encoding " , y);

#splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler();
X_train = sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)

print(X_train)
print(X_test)