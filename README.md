# 💰 Salary Prediction API 🚀

## 📝 Overview  

This project is a **Salary Prediction API** built using **FastAPI** and deployed on **AWS Lambda** ☁️. The API allows users to input their work-related details and predicts whether their salary is **above or below $50K** 💵.  

The **XGBoost model** is trained on the **Adult Income Dataset**, ensuring accuracy with **data preprocessing and class balancing**. 📊  

---

## ⭐ Features  

✅ **Web Form Interface** for user input 📝  
✅ **FastAPI** backend for real-time predictions ⚡  
✅ **XGBoost Classifier** for machine learning 🤖  
✅ **Data Preprocessing & Balancing** with `imbalanced-learn` 🔄  
✅ **AWS Lambda Deployment** for scalability ☁️  

---

## 📌 Installation  

### 🔹 Prerequisites  

Make sure you have the following installed:  

🔹 **Python 3.8+** 🐍  
🔹 **pip** (Package Manager) 📦  
🔹 **FastAPI** (Backend Framework) 🚀  
🔹 **Uvicorn** (ASGI Server) 🌍  
🔹 **Scikit-learn** (ML Utilities) 🔢  
🔹 **XGBoost** (ML Algorithm) 📈  
🔹 **Imbalanced-learn** (Class Balancing) ⚖️  
🔹 **Joblib** (Model Saving) 💾  
🔹 **Mangum** (AWS Lambda Integration) ☁️  

### 🔹 Setup  

1️⃣ Clone the repository:  
   ```sh
   git clone https://github.com/your-repo/salary-prediction-api.git
   cd salary-prediction-api
   ```
2️⃣ Install dependencies:  
   ```sh
   pip install -r requirements.txt
   ```
3️⃣ Train the model and save it:  
   ```sh
   python train_model.py
   ```

---

## 🛠 Dataset Preprocessing  

The **Adult Income Dataset** undergoes the following preprocessing steps:  

✅ **Handling Missing Values**: Replacing `?` with the most frequent value (mode)  
✅ **Label Encoding**: Converting categorical values to numerical labels  
✅ **Class Balancing**: Using **RandomOverSampler** & **RandomUnderSampler**  
✅ **Feature Scaling**: Standardizing numerical features  

---

## 🚀 Running Locally  

To start the **FastAPI** server locally, run:  

```sh
uvicorn main:app --reload
```

Then, open your browser and visit:  

```
http://127.0.0.1:8000/
```

This will load the **form UI** where users can enter their details and get predictions. 🎯  

---

## 🔗 API Endpoints  

### 📌 1. Home Page (Form UI)  

```
GET /
```

- Returns an **HTML form** for user input.  

### 📌 2. Predict Salary  

```
POST /predict/
```

- Accepts form data inputs  
- Returns JSON response with the predicted salary category  

#### 📌 Example Request:  

```sh
curl -X POST "http://127.0.0.1:8000/predict/" -d "workclass=1&education=2&occupation=3&sex=0&hours_per_week=40"
```

#### 📌 Example Response:  

```json
{
  "salary": ">50K"
}
```

---

## ☁️ Deploying on AWS Lambda  

This API is deployed on **AWS Lambda** using **Mangum**, allowing it to run via **AWS API Gateway**.  

### 🚀 Steps to Deploy  

1️⃣ Install **AWS CLI** and configure credentials:  
   ```sh
   aws configure
   ```
2️⃣ Install **Serverless Framework**:  
   ```sh
   npm install -g serverless
   ```
3️⃣ Deploy to AWS:  
   ```sh
   serverless deploy
   ```

🔹 After deployment, you will receive an **API Gateway URL** that can be used to access the API globally. 🌍  

---

## 🎯 Model Training (`train_model.py`)  

The training script performs the following steps:  

✅ Loads the dataset 📂  
✅ Preprocesses and balances data ⚖️  
✅ Trains an **XGBoost classifier** 🤖  
✅ Saves the model & scaler using `joblib` 💾  

### 🔹 Running the Training Script  

```sh
python train_model.py
```

---


