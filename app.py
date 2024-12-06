import subprocess
import sys
import streamlit as st
import os

# Function to install missing packages
def install_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            st.warning(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",
    "streamlit",
]

install_packages(required_packages)

# Import libraries after installation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit app
st.title("Wine Quality Analysis App")

# Load Dataset
uploaded_file = st.file_uploader("Upload the Wine Quality Dataset (CSV file)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    
    # Show dataset info
    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Show statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe().T)

    # Check for missing values
    st.subheader("Missing Values in the Dataset")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values[missing_values > 0])

    # Fill missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # Display histograms for numerical features
    st.subheader("Histograms of Numerical Features")
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, col in enumerate(numeric_columns):
        df[col].hist(ax=axes[idx], bins=20)
        axes[idx].set_title(col)
    for i in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sb.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Feature Engineering
    if 'quality' in df.columns:
        df['best quality'] = [1 if x > 5 else 0 for x in df['quality']]
        features = df.drop(['quality', 'best quality'], axis=1)
        target = df['best quality']
    else:
        st.error("'quality' column not found in dataset!")
        st.stop()

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

    # Train-Test Split
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=42)

    # Scale Features
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    # Model Training
    models = [LogisticRegression(), XGBClassifier(use_label_encoder=False, eval_metric='logloss'), SVC(kernel='rbf')]
    model_names = ["Logistic Regression", "XGBoost Classifier", "SVC"]
    model_accuracies = []

    st.subheader("Model Performance")
    for i, model in enumerate(models):
        model.fit(xtrain, ytrain)
        train_acc = roc_auc_score(ytrain, model.predict(xtrain))
        test_acc = roc_auc_score(ytest, model.predict(xtest))
        st.write(f"**{model_names[i]}**")
        st.write(f"- Training ROC-AUC Score: {train_acc:.2f}")
        st.write(f"- Validation ROC-AUC Score: {test_acc:.2f}")
        model_accuracies.append(test_acc)

    # Display Confusion Matrix for Best Model
    best_model_idx = model_accuracies.index(max(model_accuracies))
    st.subheader(f"The Best-Performing Model: {model_names[best_model_idx]}")

    cm = confusion_matrix(ytest, models[best_model_idx].predict(xtest))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Allow user to select a feature for histogram
    st.subheader("Feature Histogram")
    feature = st.selectbox('Select a feature for histogram:', df.columns)
    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=20, color='blue', edgecolor='black')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
