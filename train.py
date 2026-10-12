import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "dataset" / "cardio_clean.csv")

x = df.drop(columns=['cardio', 'id', 'exam_date', 'hospital_code', 'bp_reading'])
y = df['cardio']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
scaler = StandardScaler()
x_train = x_train.copy()
x_test = x_test.copy()
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#to reduce false negative
y_prob = model.predict_proba(x_test)[:, 1]
y_pred = (y_prob > 0.4).astype(int)

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:   ', recall_score(y_test, y_pred))
print('F1:       ', f1_score(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion, display_labels=['No Disease', 'Disease']).plot()
plt.title('Confusion Matrix')
plt.show()

#trained model for later use
joblib.dump(model, BASE_DIR / "model.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")