# 🍽️ Zomato Restaurant Insights & Success Predictor

A complete data analysis and machine learning web app built on the Zomato Bangalore dataset (51,000+ restaurants).

## 🚀 Live Demo
👉 [Click here to try the app](https://zomato-predictor.streamlit.app)

## 🔥 Live Features
- Interactive EDA dashboard with Plotly charts
- ML-powered restaurant success predictor
- 93%+ model accuracy using HistGradientBoosting

## 🛠️ Tech Stack
- **Python** — core language
- **Pandas & NumPy** — data cleaning & analysis
- **Matplotlib & Seaborn** — exploratory visualization
- **Scikit-learn** — machine learning pipeline
- **Streamlit & Plotly** — interactive web app

## 📁 Project Structure

```
zomato-insights/
├── app/
│   ├── app.py              ← Streamlit web app
│   ├── train_model.py      ← ML pipeline training
│   └── model.pkl           ← trained model
├── data/
│   ├── zomato.csv          ← raw dataset
│   └── clean_sample_zomato.csv
├── notebooks/
│   ├── day1_exploration.py
│   ├── day2_cleaning.py
│   ├── day3_eda.py
│   ├── day4_feature_engineering.py
│   └── day5_model.py
├── outputs/                ← saved charts
├── .gitignore
├── requirements.txt
└── README.md
```