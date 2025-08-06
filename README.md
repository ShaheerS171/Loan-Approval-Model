# 💸 Loan Approval Predictor

A smart web app that predicts whether a loan application will be **Approved** ✅ or **Not Approved** ❌ — powered by machine learning and Streamlit!

---

## 🚀 About the Project

This ML-powered app helps banks and loan officers quickly decide if a loan application should be approved, based on key applicant details like income, loan amount, education, credit history, and more.

The model was trained on real-world loan data and uses a full machine learning pipeline with:

- 🧹 Null value handling
- 🔁 Manual feature mapping (instead of encoders!)
- 📏 Feature Scaling
- ✅ Logistic Regression classifier (can swap with others)

---

## 🛠️ Tech Stack

- **Python** 🐍
- **Pandas & NumPy** – Data preprocessing
- **Scikit-learn** – ML pipeline (encoding, scaling, model)
- **Matplotlib & Seaborn** – EDA & visualizations
- **Streamlit** – For web app frontend

---

## 📊 Features

- Clean and interactive UI with Streamlit
- Predicts loan approval in real-time
- Uses saved ML model and column mapping
- Easy to modify or expand (e.g. try new models or add charts)

---

## ⚙️ How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/loan-prediction-app.git
   cd loan-prediction-app

pip install -r requirements.txt
streamlit run app.py



