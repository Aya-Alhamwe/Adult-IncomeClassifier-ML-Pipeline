import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r"C:\Users\ayaaa\Downloads\adult.csv")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Check for missing values
missing_values_percentage = df.isnull().mean() * 100
missing_values_percentage_sorted = missing_values_percentage.sort_values(ascending=False)

# Fill missing values for specific columns with the mode
for col in ['occupation', 'workclass', 'native.country']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Display categorical and numerical columns
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns

# Display value counts for income categories
result = pd.DataFrame(df['income'].value_counts(normalize=True).reset_index())
result.columns = ['income', 'norm_counts']
result['counts'] = result['norm_counts'] * len(df)

# Map income column to binary values
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# Display the count of each income class
less_50K = df[df['income'] == 0].shape[0]
more_50K = df[df['income'] == 1].shape[0]

# Normalize and display education counts
df['education'].replace(['1st-4th', '5th-6th'], 'Primary', inplace=True)
df['education'].replace(['7th-8th', '9th', '10th', '11th', '12th'], 'Middle-School', inplace=True)
df['education'].replace(['HS-grad'], 'High-School', inplace=True)
df['education'].replace(['Some-college', 'Assoc-voc', 'Assoc-acdm'], 'College', inplace=True)
df['education'].replace(['Bachelors'], 'Bachelors', inplace=True)
df['education'].replace(['Prof-school', 'Doctorate'], 'Doctorate', inplace=True)

# Normalize and display race counts
df['race'].replace(['Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], 'Others', inplace=True)

# Replace non-US native country values with "Others"
df['native.country'].loc[df['native.country'] != 'United-States'] = 'Others'

# Create a new column for capital gain minus loss
df['capital_diff'] = df['capital.gain'] - df['capital.loss']
df['capital_diff'] = pd.cut(df['capital_diff'], bins = [-5000, 5000, 100000], labels = ['Low', 'High'])
df['capital_diff'] = df['capital_diff'].astype('object')

# Drop unnecessary columns
df.drop(['capital.gain', 'capital.loss', 'fnlwgt'], axis=1, inplace=True)

# Remove outliers from hours.per.week column
df = df[~((df["hours.per.week"] > 72) | (df["hours.per.week"] < 20))]

# Split data into features (X) and target (y)
X = df.drop(columns="income")
y = df.income

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define categorical and ordinal columns for transformation
onehot_categorics = ["workclass", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]
ordinal_categorics = ["education", "capital_diff"]

# Apply transformations to categorical and ordinal columns
column_transformed = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_categorics),
        ('ordinal', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal_categorics)
    ],
    remainder='passthrough'  
)

# Fit and transform training data, transform test data
X_train_trans = column_transformed.fit_transform(X_train)
X_test_trans = column_transformed.transform(X_test)

# Handle class imbalance using oversampling and undersampling
ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)
rus = RandomUnderSampler(sampling_strategy=0.7, random_state=42)

X_resampled, y_resampled = ros.fit_resample(X_train_trans, y_train)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

# Display balanced class distribution and percentage
income_counts_resampled = pd.Series(y_resampled).value_counts()
income_percentage_resampled = (income_counts_resampled / len(y_resampled)) * 100
print("\nBalanced Class Distribution:")
print(income_counts_resampled)
print("\nBalanced Class Percentage:")
print(income_percentage_resampled)

# Convert transformed data back to DataFrame with proper column names
features = column_transformed.get_feature_names_out()
X_train = pd.DataFrame(X_train_trans, columns=features, index=X_train.index)
X_test = pd.DataFrame(X_test_trans, columns=features, index=X_test.index)

# Create and train SVC model in a pipeline with MinMaxScaler
svm_model = Pipeline([
    ("scaler", MinMaxScaler()),  # Scaling data with MinMaxScaler
    ("SVC", SVC(probability=True))  # Support Vector Classifier
])

# Fit the model to training data
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate model performance
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained model and column transformer for later use
joblib.dump(svm_model, r"C:\Users\ayaaa\Downloads\svm_income_model.pkl")
joblib.dump(column_transformed, r"C:\Users\ayaaa\Downloads\column_transformed.pkl")
