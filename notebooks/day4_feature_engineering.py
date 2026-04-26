# ============================================================
# Day 4 — Feature Engineering
# Zomato Restaurant Insights & Success Predictor
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Load Cleaned Dataset ─────────────────────────────────────
df = pd.read_csv('../data/zomato_cleaned.csv')
print(f"Dataset loaded ✅ | Shape: {df.shape}")

# ── Encode Yes/No Columns ────────────────────────────────────
df['online_order_encoded'] = df['online_order'].map({'Yes': 1, 'No': 0})
df['book_table_encoded'] = df['book_table'].map({'Yes': 1, 'No': 0})
print(f"Online order & table booking encoded ✅")

# ── Create Price Categories ──────────────────────────────────
def categorize_price(cost):
    if cost < 300:
        return 'Budget'
    elif cost < 600:
        return 'Mid-Range'
    else:
        return 'Premium'

df['price_category'] = df['approx_cost(for two people)'].apply(categorize_price)
print(f"Price categories:\n{df['price_category'].value_counts()}")

# ── Encode Location ──────────────────────────────────────────
le_location = LabelEncoder()
df['location_encoded'] = le_location.fit_transform(df['location'])
print(f"Locations encoded ✅ | Unique: {df['location'].nunique()}")

# ── Encode Restaurant Type ───────────────────────────────────
le_rest = LabelEncoder()
df['rest_type_encoded'] = le_rest.fit_transform(df['rest_type'].astype(str))
print(f"Restaurant types encoded ✅ | Unique: {df['rest_type'].nunique()}")

# ── Create Success Label ─────────────────────────────────────
df['is_successful'] = ((df['rate'] >= 4.0) & (df['votes'] >= 100)).astype(int)
print(f"\nSuccess distribution:\n{df['is_successful'].value_counts()}")
print(f"Success rate: {df['is_successful'].mean()*100:.1f}%")

# ── Prepare Final Feature Set ────────────────────────────────
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
print(f"\nFeature matrix: {X.shape}")
print(f"Target column: {y.shape}")

# ── Save Featured Dataset ────────────────────────────────────
df.to_csv('../data/zomato_featured.csv', index=False)
print("\nDay 4 Complete ✅ → data/zomato_featured.csv")