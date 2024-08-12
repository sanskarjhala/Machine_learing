## Overfitting $ undefitting 

<h3>Overfitting</h3>
<p>Overfitting occurs when our machine learning model tries to cover all the data points or more than the required data points present in the given dataset. Because of this, the model starts caching noise and inaccurate values present in the dataset, and all these factors reduce the efficiency and accuracy of the model. The overfitted model has low bias and high variance.</p>

<h3>Underfitting</h3>
<p>Underfitting occurs when our machine learning model is not able to capture the underlying trend of the data. To avoid the overfitting in the model, the fed of training data can be stopped at an early stage, due to which the model may not learn enough from the training data. As a result, it may fail to find the best fit of the dominant trend in the data.</p>

<br/>

# Polynomial Linear Regression

Polynomial linear regression is an extension of simple linear regression where the relationship between the independent variable \(X\) and the dependent variable \(Y\) is modeled as an \(n\)th-degree polynomial.

## Key Concepts

### Linear Regression Recap

In simple linear regression, the model assumes a linear relationship between the independent variable \(X\) and the dependent variable \(Y\), which is represented as:

\[
Y = \beta_0 + \beta_1 X + \epsilon
\]

Where:
- \(\beta_0\) and \(\beta_1\) are the coefficients to be estimated.
- \(\epsilon\) is the error term.

### Polynomial Regression

In polynomial regression, the model is extended to include polynomial terms of \(X\). For example:

- **Quadratic (2nd-degree) polynomial regression**:
  \[
  Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \epsilon
  \]

- **Cubic (3rd-degree) polynomial regression**:
  \[
  Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \epsilon
  \]

Here, \(\beta_2\), \(\beta_3\), etc., are the coefficients corresponding to the higher-order terms of \(X\).

### Why Polynomial Regression?

- Polynomial regression is useful when the relationship between \(X\) and \(Y\) is nonlinear but can be approximated by a polynomial function.
- It allows for more flexibility than linear regression by fitting curves instead of straight lines.

### Fitting the Model

- Polynomial regression is still a form of linear regression because the model is linear in the coefficients \(\beta_0\), \(\beta_1\), \(\beta_2\), etc.
- To fit a polynomial regression model, you can use ordinary least squares (OLS) or other methods used in linear regression.
- In practice, you can transform the original data by adding polynomial terms and then fit a linear regression model on the transformed data.

## `PolynomialFeatures` in Scikit-learn

`PolynomialFeatures` is a class provided by the `sklearn.preprocessing` module in the Scikit-learn library. It is used to generate polynomial and interaction features from a given dataset. This is particularly useful in polynomial regression, where the model involves higher-order terms of the input features.

### How `PolynomialFeatures` Works

The `PolynomialFeatures` class transforms your input data by adding polynomial combinations of the original features up to a specified degree. Here’s a breakdown of how it works:

#### Input Features

Suppose you have a dataset with a single feature \(X\). For example, if \(X = [x_1, x_2, x_3, \dots, x_n]\), the goal of `PolynomialFeatures` is to generate additional features that are powers of \(X\).

#### Polynomial Degree

You can specify the degree of the polynomial features you want to generate.

- **degree=2**: The transformed features will include:
  - The original feature: \(X\)
  - The square of the feature: \(X^2\)

- **degree=3**: The transformed features will include:
  - The original feature: \(X\)
  - The square of the feature: \(X^2\)
  - The cube of the feature: \(X^3\)

#### Interaction Features

When you have multiple input features, `PolynomialFeatures` can also generate interaction terms.

For example, if you have two features, \(X_1\) and \(X_2\), and set `degree=2`, the transformed features will include:
- \(X_1\)
- \(X_2\)
- \(X_1^2\)
- \(X_2^2\)
- The interaction term \(X_1 \times X_2\)

#### Bias (Intercept) Term

By default, `PolynomialFeatures` includes a bias (intercept) term, which is just a column of ones representing the \(X^0\) term. You can disable this by setting `include_bias=False`.


# Applying Linear Regression on Multiple Features

Linear regression is a fundamental technique in machine learning and statistics that models the relationship between one or more independent variables (features) and a dependent variable (target). When applied to multiple features, it is often referred to as "multiple linear regression."

## Key Concepts

### 1. **Understanding Multiple Linear Regression**
Multiple linear regression extends the concept of simple linear regression to include multiple independent variables. The relationship between the dependent variable \(Y\) and the independent variables \(X_1, X_2, \dots, X_n\) can be represented as:

\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
\]

Where:
- \(Y\) is the dependent variable (the outcome you want to predict).
- \(X_1, X_2, \dots, X_n\) are the independent variables (the features used for prediction).
- \(\beta_0\) is the intercept.
- \(\beta_1, \beta_2, \dots, \beta_n\) are the coefficients (weights) for each feature.
- \(\epsilon\) is the error term (the difference between the observed and predicted values).

### 2. **Data Preparation**
Before applying linear regression to multiple features, it is important to prepare your data. This involves:
- **Collecting and organizing your data**: Ensure that your dataset includes the target variable and all relevant features.
- **Handling missing values**: Missing data can skew the results, so it is crucial to fill in or remove any gaps.
- **Scaling and normalizing data**: Features with different scales can impact the performance of the model. Standardizing or normalizing the features can help in achieving better results.

### 3. **Model Assumptions**
When applying multiple linear regression, the following assumptions should be met:
- **Linearity**: The relationship between the dependent and independent variables should be linear.
- **Independence**: The observations should be independent of each other.
- **Homoscedasticity**: The variance of the errors should be constant across all levels of the independent variables.
- **No multicollinearity**: The independent variables should not be too highly correlated with each other.

### 4. **Model Fitting**
Once the data is prepared, you can proceed with fitting the linear regression model. The model will learn the best-fitting line by estimating the coefficients (\(\beta_1, \beta_2, \dots, \beta_n\)) that minimize the difference between the actual and predicted values of the dependent variable.

### 5. **Model Evaluation**
After fitting the model, it is important to evaluate its performance using various metrics such as:
- **R-squared**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
- **Residuals analysis**: Examining the residuals (the difference between observed and predicted values) can help in identifying patterns that the model did not capture.

### 6. **Making Predictions**
With the model trained and evaluated, you can use it to make predictions on new data. The input for the prediction should be in the same format as the training data, including all the features used during model training.

## Conclusion

Multiple linear regression is a powerful tool for understanding and predicting outcomes based on multiple features. By carefully preparing your data, fitting the model, and evaluating its performance, you can gain valuable insights and make accurate predictions.

For more detailed information and examples, refer to the relevant documentation and resources on linear regression.

