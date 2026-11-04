import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42

df = pd.read_csv(BASE_DIR / "dataset" / "trained_data.csv")
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

y = df["cardio"]

x = pd.get_dummies(x, drop_first=True, dtype=int)

# 2. Train-test split: 70% / 30%, stratified to keep the class ratio
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y
)

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

# 4. Hyperparameter tuning with 3-fold cross-validation (8 candidates = 24 fits).
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8, 10, 12],
    "min_samples_split": [10, 20],
    "min_samples_leaf": [5, 10],
    "max_features": ["sqrt", "log2"],
}

# Parallelism is applied only at the search level (n_jobs=-1 below);
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    param_distributions=param_grid,
    n_iter=8,
    cv=3,
    scoring="recall",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2,
)

search.fit(x_train, y_train)
model = search.best_estimator_
model.set_params(n_jobs=-1)  # use all cores for prediction

print("Best parameters:", search.best_params_)
print("Best CV recall: {:.2f}%".format(search.best_score_ * 100))

# 5. Evaluate on the held-out test set
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

confusion = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion.ravel()

print("Accuracy:  {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(
    precision_score(y_test, y_pred, zero_division=0) * 100))
print("Recall:    {:.2f}%".format(
    recall_score(y_test, y_pred, zero_division=0) * 100))
print("Specificity: {:.2f}%".format(tn / (tn + fp) * 100))
print("F1 Score:  {:.2f}%".format(
    f1_score(y_test, y_pred, zero_division=0) * 100))
print("ROC-AUC:   {:.4f}".format(roc_auc_score(y_test, y_prob)))
print("PR-AUC:    {:.4f}".format(average_precision_score(y_test, y_prob)))

# 6. Plots
ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["No Disease", "Disease"]
).plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_prob, name="Random Forest")
plt.title("Random Forest ROC Curve")
plt.show()

PrecisionRecallDisplay.from_predictions(y_test, y_prob, name="Random Forest")
plt.title("Random Forest Precision-Recall Curve")
plt.show()

importances = pd.Series(
    model.feature_importances_, index=x.columns
).sort_values()

importances.tail(10).plot(kind="barh")
plt.title("Random Forest - Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

joblib.dump(model, BASE_DIR / "random_forest_model.pkl")
joblib.dump(imputer, BASE_DIR / "random_forest_imputer.pkl")
joblib.dump(list(x.columns), BASE_DIR / "random_forest_columns.pkl")