# ============================================================
# Train Model — Full Pipeline for Streamlit App
# Zomato Restaurant Insights & Success Predictor
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
APP_DIR  = BASE_DIR

# ── Load & Clean Data ────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, 'zomato.csv'), encoding='latin-1')

df.drop(columns=['url', 'phone', 'menu_item', 'reviews_list'], inplace=True)
df.drop_duplicates(inplace=True)

df['rate'] = df['rate'].astype(str).str.extract(r'(\d+\.\d+|\d+)')
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

df['approx_cost(for two people)'] = df['approx_cost(for two people)']\
    .astype(str).str.replace(',', '', regex=False)
df['approx_cost(for two people)'] = pd.to_numeric(
    df['approx_cost(for two people)'], errors='coerce')

df['rate'] = df['rate'].fillna(df['rate'].median())
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].fillna(
    df['approx_cost(for two people)'].median())

df.dropna(subset=['location', 'rest_type', 'cuisines', 'listed_in(type)'],
          inplace=True)

print(f"Clean data shape: {df.shape}")

# ── Success Label ────────────────────────────────────────────
df['is_successful'] = ((df['rate'] >= 3.8) &
                       (df['votes'] >= 50)).astype(int)
print(f"Success rate: {df['is_successful'].mean()*100:.1f}%")

# ── Save Sample for App ──────────────────────────────────────
sample_df = df.sample(frac=0.2, random_state=42)
sample_path = os.path.join(BASE_DIR, '..', 'data', 'clean_sample_zomato.csv')
sample_df.to_csv(sample_path, index=False)
print(f"Sample saved ✅ | Shape: {sample_df.shape}")

# ── Features ─────────────────────────────────────────────────
categorical_features = [
    'online_order', 'book_table',
    'location', 'rest_type',
    'cuisines', 'listed_in(type)'
]
numerical_features = [
    'votes',
    'approx_cost(for two people)'
]

X = df[categorical_features + numerical_features]
y = df['is_successful']

# ── Pipeline ─────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        # imputer fills any remaining NaN with median before model sees data
    ]), numerical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
))
])

# ── Train ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("Model trained ✅")

# ── Evaluate ─────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# ── Save Model ───────────────────────────────────────────────
model_path = os.path.join(APP_DIR, 'model.pkl')
joblib.dump(pipeline, model_path)
print(f"Model saved ✅ → app/model.pkl")
print("\n🎉 Training Complete!")