import pandas as pd

# get the csv file & print first few records
fl = pd.read_csv("C:/Users/purnima/OneDrive/Documents/GitHub/machine-learning/00-Introduction/naive_bayes/titanic.csv")
print(fl.head())

# drop the columns of data that are not useful in prediction
fl.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis="columns", inplace=True)
print(fl.head())

#separate the input and target fields
inputs = fl.drop(["Survived"], axis="columns")
target = fl.Survived

#convert sex filed into a nominal binary field
dummies = pd.get_dummies(inputs.Sex)
print(dummies.head())

#concatenate dummies with input and drop the Sex and male fields as they are no longer required
inputs = pd.concat([inputs, dummies], axis="columns")
inputs.drop(["Sex", "male"], axis="columns", inplace=True)

#check for missing values
print(inputs.columns[inputs.isna().any()])

#missing values are found in column Age therefore, normalizing it
print(inputs.Age[:10])
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.Age[:10])

from sklearn.model_selection import train_test_split 
