#import statements

import matplotlib
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# read the given data

df = pd.read_csv("canada_per_capita_income.csv")
print(df)

# scatter plot

%matplotlib inline
plt.xlabel("year")
plt.ylabel("income per capita")
plt.scatter(df.year, df.income_per_capita, color="green", marker="*")


# extract the years from the original data

new_df = df.drop("income_per_capita", axis="columns")
print(new_df)

income_per_capita = df.income_per_capita
print(income_per_capita)

# create the linear regression object

regression = linear_model.LinearRegression()
regression.fit(new_df, income_per_capita)

# predict income per capita in the year 2020

income_per_capita_2020 = regression.predict([[2020]])
print(income_per_capita_2020)
