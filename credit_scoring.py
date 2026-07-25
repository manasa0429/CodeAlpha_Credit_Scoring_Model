
# CodeAlpha Internship - Credit Scoring Model
# Machine Learning Classification Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

# -----------------------------
# 1. Create Dataset
# -----------------------------

np.random.seed(42)

n = 1000

data = {
    'income': np.random.randint(20000, 150000, n),
    'debt': np.random.randint(5000, 100000, n),
    'credit_history_years': np.random.randint(1, 25, n),
    'number_of_late_payments': np.random.randint(0, 10, n),
    'number_of_credit_accounts': np.random.randint(1, 10, n),
    'age': np.random.randint(21, 70, n)
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Create Target Variable
# -----------------------------

score = (
    (df['income'] > 60000).astype(int)
    + (df['debt'] < 50000).astype(int)
    + (df['credit_history_years'] > 5).astype(int)
    + (df['number_of_late_payments'] < 3).astype(int)
    + (df['number_of_credit_accounts'] >= 2).astype(int)
)

df['creditworthy'] = (score >= 3).astype(int)

# -----------------------------
# 3. Split Features and Target
# -----------------------------

X = df.drop('creditworthy', axis=1)
y = df['creditworthy']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Logistic Regression
# -----------------------------

model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nLogistic Regression Performance")
print("--------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# -----------------------------
# 6. Random Forest
# -----------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("\nRandom Forest Performance")
print("-------------------------")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_roc_auc:.4f}")

# -----------------------------
# 7. Classification Report
# -----------------------------

print("\nClassification Report - Random Forest")
print("---------------------------------------")
print(classification_report(y_test, rf_pred))

# -----------------------------
# 8. Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Poor Credit', 'Good Credit'],
    yticklabels=['Poor Credit', 'Good Credit']
)

plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

plt.show()

# -----------------------------
# 9. Feature Importance
# -----------------------------

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print("------------------")
print(feature_importance)

plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title('Feature Importance in Credit Scoring')
plt.xlabel('Importance')
plt.ylabel('Financial Feature')

plt.show()

# -----------------------------
# 10. ROC Curve
# -----------------------------

lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f'Logistic Regression (AUC = {roc_auc:.2f})'
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f'Random Forest (AUC = {rf_roc_auc:.2f})'
)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()

plt.show()

# -----------------------------
# 11. Select Best Model
# -----------------------------

if rf_roc_auc > roc_auc:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_auc = rf_roc_auc
else:
    best_model = model
    best_model_name = "Logistic Regression"
    best_auc = roc_auc

print("\nBest Model:", best_model_name)
print("ROC-AUC Score:", round(best_auc, 4))

# -----------------------------
# 12. Save Model and Scaler
# -----------------------------

joblib.dump(best_model, 'credit_scoring_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel and scaler saved successfully!")
