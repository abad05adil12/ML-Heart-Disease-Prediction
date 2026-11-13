import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "dataset" / "processed" / "cardio_clean.csv"
)

df= pd.read_csv(DATA_PATH)

df= df.dropna(subset=["cardio"])

x=df.drop(
    columns=["cardio", "id", "exam_date", "hospital_code", "bp_reading"],
    errors="ignore"
)

y= df["cardio"]

#Identifying numerical and categorical columns
numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

#Preprocessing
num_pipeline= Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

category_pipeline= Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor= ColumnTransformer([
    ("numeric", num_pipeline, numeric_cols),
    ("categorical", category_pipeline, categorical_cols)
])

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

pipeline= Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

#Defining hyperparameters
param_grid= {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["lbfgs", "liblinear"],
    "model__class_weight": [None, "balanced"]
}

#Hyperparameter tuning
grid_search= GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(x_train, y_train)

#Best model
best_model= grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Recall:")
print(grid_search.best_score_)

#Test evaluation
y_pred= best_model.predict(x_test)

print()
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

OUTPUT_PATH = BASE_DIR / "tuned_logistic_regression_model.pkl"

joblib.dump(best_model, OUTPUT_PATH)

print("\nTuned Logistic Regression model saved successfully!")
