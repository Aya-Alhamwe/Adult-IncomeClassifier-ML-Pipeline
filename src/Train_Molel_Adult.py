import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\ayaaa\Downloads\adult.csv")

print(df.shape)

print(df.head())

# Convert income column to binary values
df["income"] = df["income"].map({">50K": 1, "<=50K": 0})

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

print(df.isnull().sum())

missing_values_percentage = df.isnull().mean() * 100

missing_values_percentage_sorted = missing_values_percentage.sort_values(ascending = False)

print(missing_values_percentage_sorted)


# Fill missing values with the mode
for col in ['occupation', 'workclass', 'native.country']:
    df[col].fillna(df[col].mode()[0], inplace=True)

print(df.isnull().sum())

print(df.duplicated().sum())


# Remove duplicate rows
df.drop_duplicates(inplace=True)

print(df.shape) 

#Model Train

# Encode categorical columns
label_encoder = LabelEncoder()
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship',
                       'race', 'sex', 'native.country']
df[categorical_columns] = df[categorical_columns].apply(label_encoder.fit_transform)

# Split dataset into features and target
X = df.drop(columns=['income'])
y = df['income']

# Balance the dataset using both over- and under-sampling
ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)  
rus = RandomUnderSampler(sampling_strategy=0.7, random_state=42)  
X_resampled, y_resampled = ros.fit_resample(X, y)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

# Print class distribution after balancing
income_counts_resampled = pd.Series(y_resampled).value_counts()
income_percentage_resampled = (income_counts_resampled / len(y_resampled)) * 100
print("\nBalanced Class Distribution:")
print(income_counts_resampled)
print("\nBalanced Class Percentage:")
print(income_percentage_resampled)

# Split the resampled data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Initialize and train XGBoost classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=200,  
    max_depth=6,      
    learning_rate=0.05, 
    subsample=0.8,     
    colsample_bytree=0.8,
    random_state=42
)

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('classifier', xgb_model)  
])

# Train the pipeline
pipe.fit(x_train, y_train)

# Evaluate model
test_accuracy = pipe.score(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save model and scaler
joblib.dump(pipe, r'C:\Users\ayaaa\Downloads\xgb_balanced_model.pkl')
joblib.dump(scaler, r'C:\Users\ayaaa\Downloads\scaler.pkl')
