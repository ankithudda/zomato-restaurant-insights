# ============================================================
# Day 5 — ML Model: Random Forest Classifier
# Zomato Restaurant Insights & Success Predictor
# ============================================================

import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
import pickle
import os

os.makedirs('../outputs', exist_ok=True)

# ── Load Featured Dataset ────────────────────────────────────
df = pd.read_csv('../data/zomato_featured.csv')
print(f"Dataset loaded ✅ | Shape: {df.shape}")

# ── Prepare Features ─────────────────────────────────────────
features = [
    'online_order_encoded',
    'book_table_encoded',
    'votes',
    'approx_cost(for two people)',
    'location_encoded',
    'rest_type_encoded'
]

X = df[features]
y = df['is_successful']

# ── Train Test Split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── Train Random Forest ──────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42)
rf_model.fit(X_train, y_train)
print("Model trained ✅")

# ── Evaluate Model ───────────────────────────────────────────
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Confusion Matrix ─────────────────────────────────────────
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Successful', 'Successful'],
            yticklabels=['Not Successful', 'Successful'])
plt.title('Confusion Matrix', fontsize=15)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('../outputs/confusion_matrix.png', dpi=150)
plt.show()

# ── Feature Importance ───────────────────────────────────────
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'],
         importance_df['Importance'],
         color='steelblue')
plt.title('Feature Importance', fontsize=15)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('../outputs/feature_importance.png', dpi=150)
plt.show()

# ── Save Model ───────────────────────────────────────────────
with open('../outputs/restaurant_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Model saved ✅ → outputs/restaurant_model.pkl")

print("\nDay 5 Complete ✅")
print("🎉 Full Pipeline Complete!")