# import statements

import numpy as np
import pandas as pd
from sklearn import linear_model

# read the data

df = pd.read_csv("salaries.csv")
print(df)

# replace missing values with the median

df.experience = df.experience.map({'two': 2, 'three': 3, 'five': 5, 'seven': 7, 'ten': 10, 'eleven': 11})
print(df)

df.experience.fillna(0)
print(df)

df.test_score = df.test.score.fillna(df.test_score.median())
print(df)

# build the linear regression model

model = linear_model.LinearRegression()
model.fit(df.drop("salary", axis="columns"), df.salary)

# predict salary for text cases

##################
# TEST CASES
#################
# 1. 2 years experience, 9 test score, 6 interview score
# 2. 12 year experience, 10 text score, 10 interview score

model.predict([[2, 9, 6]])
model.predict([[12, 10, 10]])