import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Status Prediction", layout="wide")

@st.cache_resource
def load_and_train_model(csv_file):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Handle missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Loan_Status'].fillna('N', inplace=True)

    # Select features like in your code
    df = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed', 'Loan_Status']]

    # Encoding
    label_self_emp = LabelEncoder()
    label_loan_status = LabelEncoder()

    df['Self_Employed'] = label_self_emp.fit_transform(df['Self_Employed'])
    df['Loan_Status'] = label_loan_status.fit_transform(df['Loan_Status'])

    # Outlier handling (IQR clipping for ApplicantIncome, LoanAmount)
    num_cols = ['ApplicantIncome', 'LoanAmount']
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)

    # Feature matrix and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define SVM models
    models = {
        'Linear': SVC(kernel='linear', C=1),
        'Polynomial': SVC(kernel='poly', degree=3, C=1),
        'RBF': SVC(kernel='rbf', C=1),
    }

    trained_models = {}
    metrics = {}

    # Train and store metrics
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        trained_models[name] = model
        metrics[name] = {'accuracy': acc, 'confusion_matrix': cm}

    # Return everything needed for inference
    return scaler, trained_models, metrics, label_self_emp, label_loan_status


def main():
    st.title("Loan Status Prediction using SVM")
    st.write(
        "Upload the loan dataset CSV, train SVM models with different kernels, "
        "and then predict loan status for a new applicant."
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload training CSV", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Training models..."):
            scaler, models, metrics, label_self_emp, label_loan_status = load_and_train_model(uploaded_file)

        # Show model performance
        st.subheader("Model Performance on Test Data")
        cols = st.columns(3)
        kernel_list = ['Linear', 'Polynomial', 'RBF']

        for i, name in enumerate(kernel_list):
            with cols[i]:
                st.markdown(f"**{name} kernel**")
                st.write(f"Accuracy: {metrics[name]['accuracy']:.4f}")

        st.subheader("Predict Loan Status for New Applicant")

        # Inputs for new prediction
        col1, col2 = st.columns(2)

        with col1:
            applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
            loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1.0)

        with col2:
            credit_history = st.selectbox("Credit History", options=[0.0, 1.0])
            self_employed_str = st.selectbox("Self Employed", options=['No', 'Yes'])

        # Use same encoding logic as training
        self_employed = 1 if self_employed_str == 'Yes' else 0

        kernel_choice = st.selectbox("Select SVM Kernel", options=kernel_list)

        if st.button("Predict"):
            model = models[kernel_choice]
            input_data = np.array([[applicant_income, loan_amount, credit_history, self_employed]])
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]

            # In your encoding, label_loan_status encodes N/Y; here we just map 1→Approved, 0→Rejected
            status = 'Approved' if pred == 1 else 'Rejected'
            st.success(f"Predicted Loan Status: {status}")
    else:
        st.info("Please upload the training CSV file to continue.")


if __name__ == "__main__":
    main()