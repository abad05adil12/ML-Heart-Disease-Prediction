import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("dataset/cardio_train_uncleaned.csv")

print(df.head())
print(df.tail())
print(df.sample(5))
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.duplicated().sum())

# Analysis of Numerical columns

df["age"].hist()
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.show()

# Outliers

sns.boxplot(x=df["weight"])
print(plt.title("Weight Distribution"))
print(plt.show())

# For blood pressure:

sns.boxplot(x=df["ap_hi"])
plt.title("Systolic Blood Pressure")
plt.show()

# For Diastolic pressure:

sns.boxplot(x=df["ap_lo"])
plt.title("Diastolic Blood Pressure")
plt.show()

# Analyze categorical variables

print(df["cholesterol"].value_counts())
sns.countplot(x="cholesterol",data=df)
plt.show()

# Analyze categorical variables

corr = df.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Analyze relationships with the target

sns.boxplot(x="cardio", y="age", data=df)
plt.show()
