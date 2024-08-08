#importing the libraries
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values
print(x)
print(y)


#splitting data 
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=1)


#include
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#predicting the test result
y_pred = regressor.predict(x_test)
print("Printing the y-predicted \n", y_pred)

#visulaising the traing data base
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train , regressor.predict(x_train) , color="blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of experince")
plt.ylabel("Salary")
plt.show()

#visulaising the test set result 
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train , regressor.predict(x_train) , color="blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of experince")
plt.ylabel("Salary")
plt.show()

#making single prediction 
print("Predicting the data for single set ")
print(regressor.predict([[13]]))

