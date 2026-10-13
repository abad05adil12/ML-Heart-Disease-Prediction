from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "dataset" / "cardio_labels_fixed.csv")

# 1. Drop unnecessary identifier/date columns first
x = df.drop(
    columns=["cardio", "id", "exam_date", "hospital_code", "bp_reading"]
)

# 2. FILL MISSING VALUES EXPLICITLY IN CODE
# For categorical columns, fill missing values with a placeholder string like 'unknown' or the mode
cat_cols = ["gender", "cholesterol", "gluc", "smoke"]
for col in cat_cols:
    if col in x.columns:
        x[col] = x[col].fillna("unknown")

# For numeric columns (if any have NaNs), fill with median
num_cols_to_check = ["age", "height", "weight", "ap_hi", "ap_lo", "alco", "active", "bmi"]
for col in num_cols_to_check:
    if col in x.columns:
        x[col] = x[col].fillna(x[col].median())

# 3. Convert categorical text columns into dummy/indicator variables
x = pd.get_dummies(x, drop_first=True)

y = df["cardio"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 4. Scale numeric columns
num_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]
scaler = StandardScaler()
x_train = x_train.copy()
x_test = x_test.copy()
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

# 5. Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# To reduce false negatives
y_prob = model.predict_proba(x_test)[:, 1]
y_pred = (y_prob > 0.4).astype(int)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1:       ", f1_score(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(
    confusion, display_labels=["NoDisease", "Disease"]
).plot()
plt.title("Confusion Matrix")
plt.show()

# Save trained model and scaler
joblib.dump(model, BASE_DIR / "model.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")
print("Model trained and saved successfully!")