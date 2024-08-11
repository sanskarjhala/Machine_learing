#importing the necessary libraires
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#importing the datasets
dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

#training the linear regression model on whole dataset 
lin_reg = LinearRegression()
lin_reg.fit(X ,y)

#visulaizing the linear regression model results 
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X) , color="yellow")
plt.title("Truth or Bulff using linear regression")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with linear regression
lin_reg_prediction = lin_reg.predict([[6.5]])
y_test = [[200000]] #suppose someone he getting $180000 in his last company 


# Optional: Plotting actual vs predicted values
plt.scatter(y_test, lin_reg_prediction)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


# Now training the polynomial regression model on the whole dataset 
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly , y)

# Visulaizing the polynomial regrtession result
plt.scatter(X,y , color="red")
plt.plot(X , lin_reg_2.predict(X_poly) , color="blue")
plt.title("truth or Bluff using polynomial regression model")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting the new result on polynomial regression model
polt_reg_prediction = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

#ploting the graph of actual value vs predicting by polynomail regression
plt.scatter(y_test , polt_reg_prediction , color="red")
plt.title("Actual vs Predict (Polynomial regression)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

