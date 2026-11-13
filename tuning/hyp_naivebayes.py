import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "dataset" / "processed" / "cardio_clean.csv"
)

df = df.dropna(subset=["cardio"])

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

numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()

categorical_cols = x.select_dtypes(
    exclude=["number"]
).columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

category_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", num_pipeline, numeric_cols),
        ("categorical", category_pipeline, categorical_cols)
    ],
    sparse_threshold=0
)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", GaussianNB())
])

param_grid = {
    "model__var_smoothing": np.logspace(-12, -6, 13)
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Recall: {:.2f}%".format(
    grid_search.best_score_ * 100
))

y_pred = best_model.predict(x_test)
y_prob = best_model.predict_proba(x_test)[:, 1]

confusion = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = confusion.ravel()

print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))

print("Precision: {:.2f}%".format(
    precision_score(y_test, y_pred, zero_division=0) * 100
))

print("Recall: {:.2f}%".format(
    recall_score(y_test, y_pred, zero_division=0) * 100
))

print("Specificity: {:.2f}%".format(
    tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
))

print("F1 Score: {:.2f}%".format(
    f1_score(y_test, y_pred, zero_division=0) * 100
))

print("ROC-AUC: {:.4f}".format(
    roc_auc_score(y_test, y_prob)
))

print("PR-AUC: {:.4f}".format(
    average_precision_score(y_test, y_prob)
))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Disease", "Disease"],
    zero_division=0
))

ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["No Disease", "Disease"]
).plot()

plt.title("Tuned Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name="Tuned Naive Bayes"
)

plt.title("Tuned Naive Bayes ROC Curve")
plt.tight_layout()
plt.show()

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob,
    name="Tuned Naive Bayes"
)

plt.title("Tuned Naive Bayes Precision-Recall Curve")
plt.tight_layout()
plt.show()

OUTPUT_PATH = BASE_DIR / "tuned_naive_bayes_model.pkl"

joblib.dump(best_model, OUTPUT_PATH)

print("Saved at:", OUTPUT_PATH)