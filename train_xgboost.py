import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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

y=df["cardio"]
x=pd.get_dummies(x, drop_first=True, dtype=int)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
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

model = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy:  {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))

print("Precision:   {:.2f}%".format(
    precision_score(y_test, y_pred, zero_division=0) *
    100
))

print("Recall:  {:.2f}%".format(
    recall_score(y_test, y_pred, zero_division=0) * 100
))

print("F1 Score:  {:.2f}%".format(
    f1_score(y_test, y_pred, zero_division=0) 
    * 100
))

confusion = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["No Disease", "Disease"]
)

display.plot()
plt.title("XGBoost Confusion Matrix")
plt.show()

joblib.dump(model, BASE_DIR / "xgboost_model.pkl")
joblib.dump(imputer, BASE_DIR / "xgboost_imputer.pkl")
joblib.dump(list(x.columns), BASE_DIR / "xgboost_columns.pkl")