# ============================================================
# Day 1 — Data Exploration
# Zomato Restaurant Insights & Success Predictor
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv('../data/zomato.csv', encoding='latin-1')
print(f"Dataset loaded ✅ | Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ── First Look ───────────────────────────────────────────────
print("\n📌 Column Names:")
print(df.columns.tolist())

print("\n📌 Data Types:")
print(df.dtypes)

# ── Missing Values ───────────────────────────────────────────
print("\n📌 Missing Values:")
print(df.isnull().sum())

missing_percent = (df.isnull().sum() / len(df)) * 100
print("\n📌 Missing Percentage:")
print(missing_percent[missing_percent > 0].round(2))

# ── Statistical Summary ──────────────────────────────────────
print("\n📌 Statistical Summary:")
print(df.describe())

# ── Chart 1: Top 10 Locations ────────────────────────────────
os.makedirs('../outputs', exist_ok=True)

plt.figure(figsize=(12, 6))
df['location'].value_counts().head(10).plot(kind='bar', color='coral')
plt.title('Top 10 Locations with Most Restaurants', fontsize=15)
plt.xlabel('Location')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../outputs/top_locations.png', dpi=150)
plt.show()

# ── Chart 2: Online Order Distribution ──────────────────────
plt.figure(figsize=(6, 6))
df['online_order'].value_counts().plot(
    kind='pie', autopct='%1.1f%%',
    colors=['lightgreen', 'salmon'], startangle=90)
plt.title('Online Order Availability', fontsize=15)
plt.ylabel('')
plt.tight_layout()
plt.savefig('../outputs/online_order_distribution.png', dpi=150)
plt.show()

# ── Chart 3: Table Booking Distribution ─────────────────────
plt.figure(figsize=(6, 6))
df['book_table'].value_counts().plot(
    kind='pie', autopct='%1.1f%%',
    colors=['steelblue', 'orange'], startangle=90)
plt.title('Table Booking Availability', fontsize=15)
plt.ylabel('')
plt.tight_layout()
plt.savefig('../outputs/book_table_distribution.png', dpi=150)
plt.show()

print("\nDay 1 Complete ✅")