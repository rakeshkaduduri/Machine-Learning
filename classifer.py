import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config("Logistic Regression", layout="centered")

# load css
def load_css(filename):
    css_path = Path(__file__).parent / filename
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title
st.markdown("""
    <div class="card">
        <h1>Logistic Regression </h1>
        <p> Predict <b> churn outcome </b> from <b> customer features </b> using Logistic Regression... </p>
    </div>
        """, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    try:
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            st.error("No CSV file found in Kaggle dataset. Check dataset structure.")
            return None
        csv_path = os.path.join(path, csv_files[0])
        data = pd.read_csv(csv_path)
        st.success(f"Loaded dataset: {csv_files[0]} with shape {data.shape}")
        return data
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}. Ensure 'kagglehub' is installed.")
        return None

df = load_data()

if df is None:
    st.stop()  # Stop execution if no data

# Dataset Preview (fixed HTML)
st.markdown('<div class="card"><h3>Dataset Preview</h3>', unsafe_allow_html=True)
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare data (moved inside if df check, but cacheable)
@st.cache_data
def prepare_model(_df):
    X = _df.drop(columns=['customerID', 'Churn'])
    y = _df['Churn'].map({'No': 0, 'Yes': 1})
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return model, scaler, X.columns, accuracy, conf_matrix, class_report

model, scaler, feature_names, accuracy, conf_matrix, class_report = prepare_model(df)

# Model Evaluation (fixed HTML)
st.markdown('<div class="card"><h3>Model Evaluation</h3>', unsafe_allow_html=True)
st.markdown(f"""
    <h4>Accuracy: {accuracy:.4f}</h4>
    <h4>Confusion Matrix:</h4>
    <pre>{conf_matrix}</pre>
    <h4>Classification Report:</h4>
    <pre>{class_report}</pre>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Confusion Matrix Visualization (fixed HTML)
st.markdown('<div class="card"><h3>Confusion Matrix Visualization</h3>', unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'], ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# About and Metrics sections (as in original)
st.markdown("""
    <div class="card">
        <h3>About Logistic Regression</h3>
        <p>Logistic Regression is a statistical method for predicting binary outcomes from data. It estimates the probability that a given input point belongs to a certain class. In this example, we predict whether a customer will churn (leave the service) based on their features.</p>  
        <p>Key steps involved:</p>
        <ul>
            <li>Data Preprocessing: Handling categorical variables and scaling features.</li>
            <li>Model Training: Fitting the Logistic Regression model to the training data.</li>
            <li>Evaluation: Assessing model performance using accuracy, confusion matrix, and classification report.</li>
        </ul>
    </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <div class="card">
        <h3>Performance Metrics Explained</h3>
        <ul>
            <li><b>Accuracy:</b> The ratio of correctly predicted observations to the total observations. It gives an overall idea of how often the model is correct.</li>
            <li><b>Confusion Matrix:</b> A table used to describe the performance of a classification model. It shows the true positives, true negatives, false positives, and false negatives.</li>
            <li><b>Classification Report:</b> Provides a detailed breakdown of precision, recall, F1-score for each class, helping to understand the model's performance on individual classes.</li>
        </ul>
    </div>
        """, unsafe_allow_html=True)

# Prediction Section (completed)
st.markdown("""
    <div class="card">
        <h3>Make Your Own Prediction</h3>
        <p>Input customer features below to predict whether they will churn or not.</p>
    </div>
        """, unsafe_allow_html=True)

# Key features for prediction (subset for simplicity; matches dataset & one-hot encoding)
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 1)
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
with col2:
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

if st.button("Predict Churn", type="primary"):
    # Create input dict matching original preprocessing (pd.get_dummies(drop_first=True))
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'gender': gender,
        'InternetService': internet_service,
        'Contract': contract
    }
    
    # Extend to full feature set with defaults (non-specified = 'No'/0)
    full_input = {name: 0 for name in feature_names}  # All dummies start 0
    full_input.update({k: input_data[k] for k in input_data})
    
    # One-hot encode matching training (only these features)
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=feature_names, fill_value=0)
    
    # Scale and predict
    input_scaled = scaler.transform(input_encoded)
    prob = model.predict_proba(input_scaled)[0, 1]
    prediction = "Yes" if prob > 0.5 else "No"
    
    st.success(f"**Churn Prediction: {prediction}**")
    st.info(f"**Churn Probability: {prob:.2%}**")