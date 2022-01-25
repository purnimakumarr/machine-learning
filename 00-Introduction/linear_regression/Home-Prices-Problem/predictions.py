import matplotlib
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("homeprices.csv")
print(df)

# create a scatter plot
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price, color = "red", marker="*")

# create a new database
new_df = df.drop("price", axis="columns")
print(new_df)

price = df.price
print(price)

# create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df, price)
reg.predict([[3300]])

reg.coef_ # prints coefficient
reg.intercept_ # prints intercept

reg.predict([[5000]])

# generate csv file for home price predictions
area_df = pd.read_csv("areas.csv")
print(area_df.head(3))

p = reg.predict(area_df)
print(p)

# add new column for prdicted home price
area_df["prices"] = p
print(area_df)

area_df.to_csv("prediction.csv")


