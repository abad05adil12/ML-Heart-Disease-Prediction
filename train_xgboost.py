
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent

# 1. Load dataset
df = pd.read_csv(BASE_DIR / "dataset" / "trained_data.csv")

df = df.dropna(subset=["cardio"])

# 2. Select features and target
x = df.drop(
    columns=[
        "cardio",
        "id",
        "exam_date",
        "hospital_code",
        "bp_reading"
    ],
    errors="ignore"
)

y = df["cardio"].astype(int)

# 3. Encode categorical columns
x = pd.get_dummies(x, drop_first=True, dtype=int)

# 4. Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Handle missing values using training data only
imputer = SimpleImputer(strategy="median")

x_train = pd.DataFrame(
    imputer.fit_transform(x_train),
    columns=x_train.columns,
    index=x_train.index
)

x_test = pd.DataFrame(
    imputer.transform(x_test),
    columns=x_test.columns,
    index=x_test.index
)

# 6. Calculate class balance from training data
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_weight = (
    negative_count / positive_count
    if positive_count > 0
    else 1.0
)

# 7. Create XGBoost model
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

# 8. Define hyperparameters to test
param_grid = {
    "n_estimators": [200, 300, 500, 700],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 2, 5, 10],
    "scale_pos_weight": [1, scale_weight]
}

# 9. Tune model using training data only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

search.fit(x_train, y_train)

# 10. Get the best model
best_model = search.best_estimator_

# 11. Predict on the untouched test set
y_pred = best_model.predict(x_test)

# 12. Display best parameters
print("\nBest Parameters:")
print(search.best_params_)

print(
    "\nBest Cross-Validation F1 Score: {:.2f}%".format(
        search.best_score_ * 100
    )
)

# 13. Evaluate final model
print("\n===== TUNED XGBOOST RESULTS =====")

print("Accuracy:  {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))

print("Precision: {:.2f}%".format(
    precision_score(y_test, y_pred, zero_division=0) * 100
))

print("Recall:    {:.2f}%".format(
    recall_score(y_test, y_pred, zero_division=0) * 100
))

print("F1 Score:  {:.2f}%".format(
    f1_score(y_test, y_pred, zero_division=0) * 100
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Disease", "Disease"],
    zero_division=0
))

# 14. Confusion matrix
confusion = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["No Disease", "Disease"]
)

display.plot()
plt.title("Tuned XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# 15. Save the best model and supporting files
joblib.dump(
    best_model,
    BASE_DIR / "xgboost_model.pkl"
)

joblib.dump(
    imputer,
    BASE_DIR / "xgboost_imputer.pkl"
)

joblib.dump(
    list(x.columns),
    BASE_DIR / "xgboost_columns.pkl"
)

print("\nBest XGBoost model saved successfully.")
