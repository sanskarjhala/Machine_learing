# Assumptions of Linear regression

<h2>Mainly there are 7 assumptions taken while using Linear Regression:</h2>
<br/>

<h4>1. Linear Model</h4>
<p>According to this assumption, the relationship between the independent and dependent variables should be linear. The reason behind this relationship is that if the relationship will be non-linear which is certainly is the case in the real-world data then the predictions made by our linear regression model will not be accurate and will vary from the actual observations a lot.</p>

<h4>2. No Multicolinearlity in the data</h4>
<p>If the predictor variables are correlated among themselves, then the data is said to have a multicollinearity problem. But why is this a problem? The answer to this question is that high collinearity means that the two variables vary very similarly and contain the same kind of information. This will leads to redundancy in the dataset. Due to redundancy, only the complexity of the model increase, and no new information or pattern is learned by the model. We generally try to avoid highly correlated features even while using complex models.</p>

<h4>3. Homoscedasticity of Residuals or Equal Variances</h4>
<p>Homoscedasity is the term that states that the spread residuals which we are getting from the linear regression model should be homogeneous or equal spaces. If the spread of the residuals is heterogeneous then the model is called to be an unsatisfactory model.</p>
<img src="https://media.geeksforgeeks.org/wp-content/uploads/20221104123858/SatisfactoryUnsatisfactoryModel.png" alt="picture represting Equal variance" >


<h4>4. No Autocorrelation in residuals</h4>
<p>One of the critical assumptions of multiple linear regression is that there should be no autocorrelation in the data. When the residuals are dependent on each other, there is autocorrelation. This factor is visible in the case of stock prices when the price of a stock is not independent of its previous one.</p>


<h4>5. Number of observations Greater than the number of predictors</h4>
<p>For a better-performing model, the number of training data or observations should be always greater than the number of test or prediction data. However greater the number of observations better the model performance. Therefore, to build a linear regression model you must have more observations than the number of independent variables (predictors) in the data set. The reason behind this can be understood by the curse of dimensionality.</p>


<h4>6. Each observation is unique</h4>
<p>It is also important to ensure that each observation is independent of the other observation.  Meaning each observation in the data set should be measured separately on a unique occurrence of the event that caused the observation.</p>

<h4>7. the Oulier Check</h4>
<br/>

<h3>Dummy variable trap</h3>
<p>The Dummy variable trap is a scenario where there are attributes that are highly correlated (Multicollinear) and one variable predicts the value of others. When we use one-hot encoding for handling the categorical data, then one dummy variable (attribute) can be predicted with the help of other dummy variables. Hence, one dummy variable is highly correlated with other dummy variables. Using all dummy variables for regression models leads to a dummy variable trap. So, the regression models should be designed to exclude one dummy variable. </p>