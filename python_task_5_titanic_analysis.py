import pandas as pd

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Fill missing ages
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Filter adults with fare > 30
filtered_df = df[(df["Age"] >= 18) & (df["Fare"] > 30)]

print("Adults with Fare > 30")
print(filtered_df.head(10))

# New Features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

print("\nNew Features")
print(df[["FamilySize", "FarePerPerson"]].head(10))

# Average fare by class
print("\nAverage Fare by Class")
print(df.groupby("Pclass")["Fare"].mean())

# Survival rate by class
print("\nSurvival Rate by Class")
print(df.groupby("Pclass")["Survived"].mean())

# Survival rate by gender
print("\nSurvival Rate by Gender")
print(df.groupby("Sex")["Survived"].mean())