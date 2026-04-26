# ============================================================
# Day 2 — Data Cleaning
# Zomato Restaurant Insights & Success Predictor
# ============================================================

import pandas as pd
import numpy as np

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv('../data/zomato.csv', encoding='latin-1')
print(f"Original Shape: {df.shape}")

# ── Drop Useless Columns ─────────────────────────────────────
df.drop(columns=['url', 'phone', 'menu_item', 'reviews_list'], inplace=True)
print(f"After dropping columns: {df.shape}")

# ── Remove Duplicates ────────────────────────────────────────
print(f"Duplicates found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"After removing duplicates: {df.shape}")

# ── Clean Rate Column ────────────────────────────────────────
df['rate'] = df['rate'].astype(str).str.extract(r'(\d+\.\d+|\d+)')
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
print(f"Rate dtype: {df['rate'].dtype}")

# ── Clean approx_cost Column ────────────────────────────────
df['approx_cost(for two people)'] = df['approx_cost(for two people)']\
    .astype(str).str.replace(',', '', regex=False)
df['approx_cost(for two people)'] = pd.to_numeric(
    df['approx_cost(for two people)'], errors='coerce')
print(f"Cost dtype: {df['approx_cost(for two people)'].dtype}")

# ── Handle Missing Values ────────────────────────────────────
df['rate'].fillna(df['rate'].median(), inplace=True)
df['approx_cost(for two people)'].fillna(
    df['approx_cost(for two people)'].median(), inplace=True)
df.dropna(subset=['location', 'rest_type', 'cuisines'], inplace=True)
print(f"Final Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# ── Save Cleaned Dataset ─────────────────────────────────────
df.to_csv('../data/zomato_cleaned.csv', index=False)
print("\nDay 2 Complete ✅ → data/zomato_cleaned.csv")