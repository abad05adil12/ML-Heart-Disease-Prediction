import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
    classification_report,confusion_matrix
)
BASE_DIR = Path(__file__).resolve().parent.parent

df=pd.read_csv(
    BASE_DIR / "dataset" / "processed" / "cardio_clean.csv"
)

df = df.dropna(subset=["cardio"])

#Separate features and target
x=df.drop(
    columns=["cardio", "id", "exam_date", "hospital_code", "bp_reading"],
    errors="ignore"
)

y = df["cardio"].astype(int)

#Identifying numerical and categorical columns
numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

#Preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

category_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", num_pipeline, numeric_cols),
    ("categorical", category_pipeline, categorical_cols)
])

#Train-test split
x_train, x_test, y_train, y_test= train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

#Creating pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))])

#Defining hyperparameters
param_grid= {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [5, 10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 5]
}

#Hyperparameter tuning
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(x_train, y_train)

best_model= grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Recall: {:.2f}%".format(
    grid_search.best_score_ * 100
))

y_pred=best_model.predict(x_test)

print("\nFinal Decision Tree Results")

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

print("Precision: {:.2f}%".format(precision_score(y_test, y_pred, zero_division=0) * 100))

print("Recall: {:.2f}%".format(recall_score(y_test, y_pred, zero_division=0) * 100))

print("F1: {:.2f}%".format(f1_score(y_test, y_pred, zero_division=0) * 100))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

OUTPUT_PATH= BASE_DIR / "tuned_decision_tree_model.pkl"

joblib.dump(best_model, OUTPUT_PATH)
