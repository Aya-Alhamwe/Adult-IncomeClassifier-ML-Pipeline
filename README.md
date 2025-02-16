# 💰 Salary Prediction Web Application 🌟

🚀 **This application is deploy and live on Render!**

This project is a web application built with FastAPI that predicts whether a user's salary is above or below $50K💵 based on work-related details.

## 🚀 Features
- 🖥️ Interactive web interface for user input.
- 🤖 Machine learning model using XGBoost for salary prediction.
- ⚙️ Data preprocessing with StandardScaler.
- ☁️ Deployment-ready with Render.

## 🛠️ Installation

1. **🔍 Clone the repository:**
```bash
git clone <repository_url>
cd <repository_name>
```

2. **📦 Install dependencies:**
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
fastapi
uvicorn
joblib
pandas
scikit-learn
xgboost
python-multipart
```

## ⚙️ How to Run

1. **🚀 Start the FastAPI application:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. **🌐 Access the web interface:**
- Open [http://localhost:8000](http://localhost:8000) in your browser.

## 🧠 Model Training

- The `train_model.py` file loads the `adult.csv` dataset, preprocesses it, balances the classes, trains an XGBoost model, and saves the model and scaler using `joblib`.

### 🛠️ Steps:
1. **📊 Prepare Dataset:** Missing values are replaced with the mode.
2. **🔠 Encode Categorical Features:** Label encoding is applied.
3. **⚖️ Balance Dataset:** Over-sampling and under-sampling techniques are used.
4. **🏋️ Model Training:** XGBoost with optimized hyperparameters.
5. **💾 Save Artifacts:** Model and scaler are saved as `xgb_balanced_model.pkl` and `scaler.pkl`.

## 🔍 Usage Guide
- 🧑‍💼 Select workclass, education, occupation, and sex.
- ⏱️ Enter hours per week.
- 🎯 Click "Predict Salary" to get the prediction.

## 📦 Deployment on Render

1. **⚙️ Create a new Render web service.**
2. **🛠️ Use the following build command:**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```
3. **🔧 Ensure the following environment variables are set:**
- `PYTHON_VERSION` to match your environment.

4. **🚀 Deploy and access the web app from the provided URL.**

### 🌐 Live Application
- The application is successfully deployed on Render and can be accessed here: [Salary Predictor](https://adultproject.onrender.com)

## ⚠️ Important Notes
- 🗂️ Ensure `xgb_balanced_model.pkl` and `scaler.pkl` are present in the root directory.
- 📁 Adjust the file paths if necessary.
- 🖼️ The app uses basic dropdowns and numeric inputs; consider enhancing for better UX.

🎯 **Enjoy Predicting Salaries!** 💼💵
