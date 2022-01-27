# Aim: Create a Polynomial Regression Model(degree = 2) & Linear Regression Model for the same data set and compare the two models by testing on the record => (Level = 6.5) and (Level = 3.5)
# see predictions.ipynb for output

# import statements

import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model

# read data file

df = pd.read_csv("Position_Salaries.csv")
print(df)

# drop all unnecessary columns

new_df = df.drop("Position", axis="columns")
print(new_df)

# scatter plot the data points

import matplotlib.pyplot as plt
%matplotlib inline

plt.xlabel('Level')
plt.ylabel('Salary')
plt.scatter(df.Level, df.Salary, color = "red", marker=".")

# create a linear regression model and a polynomial regression model

polynomial_reg = preprocessing.PolynomialFeatures(degree = 2)
linear_reg = linear_model.LinearRegression()
linear_reg2 = linear_model.LinearRegression()

level = new_df.drop("Salary", axis="columns")
salary = new_df.drop("Level", axis="columns")

polynomial = polynomial_reg.fit_transform(level)
linear_reg.fit(polynomial, salary)

linear_reg2.fit(level, salary)

# scatter plot linear model and polynomial model so as to compare the training error

plt.xlabel("Level")
plt.ylabel("Salary")
plt.scatter(level, salary, color = "red")
plt.plot(level, linear_reg2.predict(level), color = "blue") # linear model
plt.plot(level, linear_reg.predict(polynomial), color = "green") # polynomial model

# predict salary for level 6.5

linear_reg2.predict([[6.5]])
linear_reg.predict(polynomial_reg.fit_transform([[6.5]]))

# Conclusion: Polynomial Model has less predition error as compared to Linear Model