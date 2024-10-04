import pandas as pd

df = pd.read_csv("titanic.csv")
# First five rows
print("First five rows")
print(df.head())
# Summary
print("Summary Statistics")
print(df.describe())
# Filling NaN values
print("Filling NaN values")
print(df.fillna("Not Available"))
# Grouping data by survived
print("Survivor Data")
print(df.groupby("Survived")[["Age","Fare"]].mean())