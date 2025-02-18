# 💰 Income Prediction Model with FastAPI & Machine Learning 🚀

## 🌟 Overview
I developed this project to analyze data and predict income using machine learning. The project includes extensive data analysis, a web interface built with FastAPI, and model deployment on Render.

🔗 [🔗 Render App Link](https://adult-incomeclassifier-ml-pipeline.onrender.com)

---

## ⚙️ Features
- 🛠️ **Feature Engineering**: Performed thorough data preprocessing, including:
  - Replacing missing values.
  - Simplifying and unifying categorical variables.
  - Creating new features like "capital_diff".
- 📊 **Exploratory Data Analysis (EDA)**: Conducted in-depth data analysis with visualizations to uncover patterns and trends.
- 🎛️ **Feature Scaling & Encoding**: Applied OneHotEncoder, OrdinalEncoder, and MinMaxScaler for efficient data preparation.
- ⚖️ **Data Balancing**: The dataset was imbalanced, so I experimented with **SMOTE**, but the best results were achieved using **Random OverSampling** and **Random UnderSampling**.
- 🔍 **Model Experimentation**: Tested various models like Decision Trees, Random Forest, and XGBoost, but the **SVM model delivered the best performance**.
- ⚡ **Hyperparameter Tuning**: Applied hyperparameter optimization to improve model performance.
- 🌐 **FastAPI Integration**: Integrated the model into a web application using FastAPI.
- 🚀 **Deployment**: Deployed the application on Render for easy access.

---

## 🔢 Data Insights
- The dataset is derived from the Adult Income dataset.
- Extensive data cleaning, missing value handling, and feature engineering were performed to identify key factors influencing income.

---

## 🛠️ Tech Stack
- 🐍 Python (Pandas, NumPy, Scikit-Learn, XGBoost, imbalanced-learn)
- 📊 Seaborn, Matplotlib
- ⚙️ FastAPI
- 🌐 Render (Deployment)

---

## 🚀 How to Run Locally
```bash
# Clone the repo
git clone https://github.com/YourUsername/Income-Prediction-FastAPI.git
cd Income-Prediction-FastAPI

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app:app --reload
```

Then navigate to `http://127.0.0.1:8000` to interact with the interface.

---

## 📈 Sample Visualizations
🔹 Visual analyses of educational levels, racial distributions, and financial patterns.
🔹 Plots illustrating class balance improvements after resampling.




