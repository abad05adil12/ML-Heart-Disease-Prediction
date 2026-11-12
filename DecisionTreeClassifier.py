import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from recommendations import generate_suggestions

BASE_DIR = Path(__file__).resolve().parent

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
    random_state=42,
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

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

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

ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["No Disease", "Disease"]
).plot()

plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name="Decision Tree"
)

plt.title("Decision Tree ROC Curve")
plt.tight_layout()
plt.show()

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob,
    name="Decision Tree"
)

plt.title("Decision Tree Precision-Recall Curve")
plt.tight_layout()
plt.show()

patient_data = x_test.iloc[[0]]

prediction = model.predict(patient_data)[0]

# Replace these example values with original, unscaled
# measurements belonging to the same patient.
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

print("\nNote: This prediction is not a medical diagnosis.")

# Save Model
joblib.dump(
    model,
    BASE_DIR / "decision_tree_model.pkl"
)

joblib.dump(
    imputer,
    BASE_DIR / "decision_tree_imputer.pkl"
)

joblib.dump(
    list(x.columns),
    BASE_DIR / "decision_tree_columns.pkl"
)
