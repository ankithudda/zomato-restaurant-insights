# ============================================================
# Day 3 — Exploratory Data Analysis
# Zomato Restaurant Insights & Success Predictor
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../outputs', exist_ok=True)
sns.set_style('whitegrid')

# ── Load Cleaned Dataset ─────────────────────────────────────
df = pd.read_csv('../data/zomato_cleaned.csv')
print(f"Dataset loaded ✅ | Shape: {df.shape}")

# ── Chart 1: Rating Distribution ────────────────────────────
plt.figure(figsize=(10, 5))
sns.histplot(df['rate'], bins=20, kde=True, color='steelblue')
plt.title('Distribution of Restaurant Ratings', fontsize=15)
plt.xlabel('Rating')
plt.ylabel('Number of Restaurants')
plt.tight_layout()
plt.savefig('../outputs/rating_distribution.png', dpi=150)
plt.show()

# ── Chart 2: Online Order vs Rating ─────────────────────────
plt.figure(figsize=(8, 5))
sns.boxplot(x='online_order', y='rate', data=df, palette='Set2')
plt.title('Online Order vs Rating', fontsize=15)
plt.xlabel('Online Order Available')
plt.ylabel('Rating')
plt.tight_layout()
plt.savefig('../outputs/online_order_vs_rating.png', dpi=150)
plt.show()

# ── Chart 3: Table Booking vs Rating ────────────────────────
plt.figure(figsize=(8, 5))
sns.boxplot(x='book_table', y='rate', data=df, palette='Set1')
plt.title('Table Booking vs Rating', fontsize=15)
plt.xlabel('Table Booking Available')
plt.ylabel('Rating')
plt.tight_layout()
plt.savefig('../outputs/book_table_vs_rating.png', dpi=150)
plt.show()

# ── Chart 4: Top 10 Cuisines ─────────────────────────────────
plt.figure(figsize=(12, 6))
df['cuisines'].value_counts().head(10).plot(kind='barh', color='mediumpurple')
plt.title('Top 10 Most Popular Cuisines', fontsize=15)
plt.xlabel('Number of Restaurants')
plt.ylabel('Cuisine Type')
plt.tight_layout()
plt.savefig('../outputs/top_cuisines.png', dpi=150)
plt.show()

# ── Chart 5: Cost vs Rating ──────────────────────────────────
plt.figure(figsize=(10, 5))
sns.scatterplot(x='approx_cost(for two people)',
                y='rate', data=df,
                alpha=0.4, color='tomato')
plt.title('Cost vs Rating', fontsize=15)
plt.xlabel('Approx Cost for Two (₹)')
plt.ylabel('Rating')
plt.tight_layout()
plt.savefig('../outputs/cost_vs_rating.png', dpi=150)
plt.show()

# ── Chart 6: Top 10 Restaurant Types ────────────────────────
plt.figure(figsize=(12, 6))
df['rest_type'].value_counts().head(10).plot(kind='barh', color='darkorange')
plt.title('Top 10 Restaurant Types', fontsize=15)
plt.xlabel('Number of Restaurants')
plt.ylabel('Restaurant Type')
plt.tight_layout()
plt.savefig('../outputs/top_rest_types.png', dpi=150)
plt.show()

print("\nDay 3 Complete ✅")