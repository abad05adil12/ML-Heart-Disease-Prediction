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

from recommendations import generate_suggestions

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

y = df["cardio"].astype(int)

x = pd.get_dummies(x, drop_first=True, dtype=int)

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

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8, 10, 12],
    "min_samples_split": [10, 20],
    "min_samples_leaf": [5, 10],
    "max_features": ["sqrt", "log2"],
}

search = RandomizedSearchCV(
    RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1
    ),
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
model.set_params(n_jobs=-1)

print("Best parameters:", search.best_params_)
print("Best CV recall: {:.2f}%".format(
    search.best_score_ * 100
))

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

confusion = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = confusion.ravel()

print("Accuracy:  {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))

print("Precision: {:.2f}%".format(
    precision_score(y_test, y_pred, zero_division=0) * 100
))

print("Recall:    {:.2f}%".format(
    recall_score(y_test, y_pred, zero_division=0) * 100
))

print("Specificity: {:.2f}%".format(
    tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
))

print("F1 Score:  {:.2f}%".format(
    f1_score(y_test, y_pred, zero_division=0) * 100
))

print("ROC-AUC:   {:.4f}".format(
    roc_auc_score(y_test, y_prob)
))

print("PR-AUC:    {:.4f}".format(
    average_precision_score(y_test, y_prob)
))

ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["No Disease", "Disease"]
).plot()

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name="Random Forest"
)

plt.title("Random Forest ROC Curve")
plt.tight_layout()
plt.show()

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob,
    name="Random Forest"
)

plt.title("Random Forest Precision-Recall Curve")
plt.tight_layout()
plt.show()

importances = pd.Series(
    model.feature_importances_,
    index=x.columns
).sort_values()

importances.tail(10).plot(kind="barh")

plt.title("Random Forest - Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

prediction = model.predict(x_test.iloc[[0]])[0]

patient = {
    "ap_hi": 160,
    "cholesterol": 3,
    "gluc": 2,
    "bmi": 31
}

result, suggestions = generate_suggestions(
    patient,
    prediction
)
print("Prediction:", result)

print("\nPersonalized Suggestions:")

for suggestion in suggestions:
    print("-", suggestion)

joblib.dump(
    model,
    BASE_DIR / "random_forest_model.pkl"
)

joblib.dump(
    imputer,
    BASE_DIR / "random_forest_imputer.pkl"
)

joblib.dump(
    list(x.columns),
    BASE_DIR / "random_forest_columns.pkl"
)
