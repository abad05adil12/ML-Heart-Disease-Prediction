import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# df=pd.read_csv("dataset/cardio_train_uncleaned.csv")
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "cardio_labels_fixed.csv"

df = pd.read_csv(DATA_PATH)

print(df.head())
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

# df["age"].hist()
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.title("Distribution of Age")
# plt.show()

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

#Removing Missing Values
#print(df.isnull().sum())
df['id']=df['id'].fillna(df['id'].mean())
df['gender']=df['gender'].fillna(df['gender'].mode()[0])
df['height']=df['height'].fillna(df['height'].mean())
df['weight']=df['weight'].fillna(df['weight'].mean())
df['ap_hi']=df['ap_hi'].fillna(df['ap_hi'].mode()[0])
df['ap_lo']=df['ap_lo'].fillna(df['ap_lo'].mode()[0])
df['cholesterol']=df['cholesterol'].fillna(df['cholesterol'].mode()[0])
df['gluc']=df['gluc'].fillna(df['gluc'].mode()[0])
df['smoke']=df['smoke'].fillna(df['smoke'].mode()[0])
df['alco']=df['alco'].fillna(df['alco'].mode()[0])
df['active']=df['active'].fillna(df['active'].mode()[0])
df['cardio']=df['cardio'].fillna(df['cardio'].mean())
df['exam_date']=df['exam_date'].fillna(df['exam_date'].mode()[0])
df['hospital_code']=df['hospital_code'].fillna(df['hospital_code'].mode()[0])
df['bp_reading']=df['bp_reading'].fillna(df['bp_reading'].mode()[0])

#print(df.isnull().sum())

print()

print('Before Removing Duplicates')
print(df.duplicated().sum())

print('After Removing Duplicates')
df=df.drop_duplicates()
print(df.duplicated().sum())
print()

#To Check how many rows got ? and unknown
#print((df.astype(str).apply(
    #lambda row: row.str.lower().isin(['?', 'unknown']).any(),
    #axis=1
#)).sum())

df = df.replace(["?", "unknown"], np.nan)
df = df.dropna()

print("Remaining ? and unknown values:", df.isin(["?", "unknown"]).sum().sum())

le=LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['cholesterol'] = le.fit_transform(df['cholesterol'])
df['gluc'] = le.fit_transform(df['gluc'])


scaler = StandardScaler()
df[['age','height','weight','ap_hi','ap_lo']]=scaler.fit_transform(
    df[['age','height','weight','ap_hi','ap_lo']]
)

print(df[['age','height','weight','ap_hi','ap_lo']].head())
