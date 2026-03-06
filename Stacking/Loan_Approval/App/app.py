# import streamlit as st
# import numpy as np
# import pandas as pd

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # -----------------------------
# # PAGE CONFIG
# # -----------------------------
# st.set_page_config(
#     page_title="Smart Loan Approval System",
#     page_icon="🎯",
#     layout="centered"
# )

# # -----------------------------
# # APP TITLE & DESCRIPTION
# # -----------------------------
# st.title("🎯 Smart Loan Approval System – Stacking Model")

# st.write(
#     """
#     This system uses a **Stacking Ensemble Machine Learning model**  
#     to predict whether a loan will be **Approved or Rejected** by  
#     combining multiple ML models for better decision making.
#     """
# )

# # -----------------------------
# # LOAD & PREPARE DATA
# # -----------------------------
# @st.cache_data
# def load_and_train_model():
#     df = pd.read_csv("train_Loan_Prediction.csv")  # ensure file exists

#     # Drop ID
#     df.drop(columns=['Loan_ID'], inplace=True)

#     # Encoding
#     df.replace({
#         'Gender': {'Male': 1, 'Female': 0},
#         'Married': {'Yes': 1, 'No': 0},
#         'Education': {'Graduate': 1, 'Not Graduate': 0},
#         'Self_Employed': {'Yes': 1, 'No': 0},
#         'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
#         'Loan_Status': {'Y': 1, 'N': 0}
#     }, inplace=True)

#     df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

#     # Fill missing values
#     df.fillna(df.median(numeric_only=True), inplace=True)

#     X = df.drop('Loan_Status', axis=1)
#     y = df['Loan_Status']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)

#     # Base models
#     base_models = [
#         ('lr', LogisticRegression(max_iter=1000)),
#         ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
#         ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
#     ]

#     meta_model = LogisticRegression()

#     stacking_model = StackingClassifier(
#         estimators=base_models,
#         final_estimator=meta_model,
#         cv=5
#     )

#     stacking_model.fit(X_train_scaled, y_train)

#     return stacking_model, scaler, base_models


# model, scaler, base_models = load_and_train_model()

# # -----------------------------
# # SIDEBAR INPUT SECTION
# # -----------------------------
# st.sidebar.header("📝 Applicant Details")

# app_income = st.sidebar.number_input("Applicant Income", min_value=0)
# co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
# loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
# loan_term = st.sidebar.number_input("Loan Amount Term (Months)", min_value=0)

# credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
# credit_history = 1 if credit_history == "Yes" else 0

# employment = st.sidebar.selectbox(
#     "Employment Status", ["Salaried", "Self-Employed"]
# )
# self_employed = 0 if employment == "Salaried" else 1

# property_area = st.sidebar.selectbox(
#     "Property Area", ["Urban", "Semi-Urban", "Rural"]
# )
# property_area_map = {"Rural": 0, "Semi-Urban": 1, "Urban": 2}
# property_area = property_area_map[property_area]

# # -----------------------------
# # MODEL ARCHITECTURE DISPLAY
# # -----------------------------
# st.subheader("🧩 Model Architecture (Stacking Ensemble)")

# st.markdown(
#     """
# **Base Models Used:**
# - Logistic Regression
# - Decision Tree
# - Random Forest

# **Meta Model Used:**
# - Logistic Regression

# 📌 Predictions from base models are combined and passed to the meta-model  
# to make the final loan approval decision.
# """
# )

# # -----------------------------
# # PREDICTION BUTTON
# # -----------------------------
# if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

#     input_data = np.array([[
#         app_income,
#         co_income,
#         loan_amount,
#         loan_term,
#         credit_history,
#         self_employed,
#         property_area
#     ]])

#     input_scaled = scaler.transform(input_data)

#     # Base model predictions
#     base_predictions = {}
#     for name, model_base in base_models:
#         model_base.fit(scaler.transform(model.estimators_[0][1].X_), model.estimators_[0][1].y_)
#         pred = model_base.predict(input_scaled)[0]
#         base_predictions[name] = pred

#     final_prediction = model.predict(input_scaled)[0]
#     confidence = model.predict_proba(input_scaled)[0][final_prediction] * 100

#     # -----------------------------
#     # OUTPUT SECTION
#     # -----------------------------
#     st.subheader("📌 Prediction Result")

#     if final_prediction == 1:
#         st.success("✅ Loan Approved")
#     else:
#         st.error("❌ Loan Rejected")

#     st.markdown("### 📊 Base Model Predictions")
#     st.write(f"**Logistic Regression:** {'Approved' if base_predictions['lr'] else 'Rejected'}")
#     st.write(f"**Decision Tree:** {'Approved' if base_predictions['dt'] else 'Rejected'}")
#     st.write(f"**Random Forest:** {'Approved' if base_predictions['rf'] else 'Rejected'}")

#     st.markdown("### 🧠 Final Stacking Decision")
#     st.write("**Approved**" if final_prediction == 1 else "**Rejected**")

#     st.markdown(f"### 📈 Confidence Score")
#     st.write(f"{confidence:.2f}%")

#     # -----------------------------
#     # BUSINESS EXPLANATION
#     # -----------------------------
#     st.subheader("💼 Business Explanation")

#     explanation = (
#         "Based on applicant income, credit history, and combined predictions "
#         "from multiple machine learning models, the applicant is "
#         f"{'likely' if final_prediction == 1 else 'unlikely'} to repay the loan. "
#         f"Therefore, the stacking model predicts loan "
#         f"{'approval' if final_prediction == 1 else 'rejection'}."
#     )

#     st.info(explanation)

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🎯",
    layout="centered"
)

# ==================================================
# TITLE & DESCRIPTION
# ==================================================
st.title("🎯 Smart Loan Approval System – Stacking Model")

st.markdown("""
This system uses a **Stacking Ensemble Machine Learning model**  
to predict whether a **loan will be Approved or Rejected**  
by combining multiple ML models for better decision making.
""")

# ==================================================
# LOAD DATA & TRAIN MODELS
# ==================================================
@st.cache_data
def load_and_train_model():

    # Load dataset
    df = pd.read_csv("train_Loan_Prediction.csv")

    # Drop ID column
    df.drop(columns=["Loan_ID"], inplace=True)

    # Encode categorical variables
    df.replace({
        "Gender": {"Male": 1, "Female": 0},
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
        "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
        "Loan_Status": {"Y": 1, "N": 0}
    }, inplace=True)

    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Features & target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Base models
    lr = LogisticRegression(max_iter=1000)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train base models
    lr.fit(X_train_scaled, y_train)
    dt.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    # Stacking model
    stacking_model = StackingClassifier(
        estimators=[
            ("Logistic Regression", lr),
            ("Decision Tree", dt),
            ("Random Forest", rf)
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )

    stacking_model.fit(X_train_scaled, y_train)

    return stacking_model, scaler, lr, dt, rf


model, scaler, lr, dt, rf = load_and_train_model()

# ==================================================
# SIDEBAR INPUTS
# ==================================================
st.sidebar.header("📝 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term (Months)", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
credit_history = 1 if credit_history == "Yes" else 0

employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
self_employed = 0 if employment == "Salaried" else 1

property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
property_area_map = {"Rural": 0, "Semi-Urban": 1, "Urban": 2}
property_area = property_area_map[property_area]

# ==================================================
# MODEL ARCHITECTURE DISPLAY
# ==================================================
st.subheader("🧠 Model Architecture (Stacking Ensemble)")

st.markdown("""
**Base Models Used:**
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

**Meta Model Used:**
- Logistic Regression  

📌 Predictions from base models are combined and passed to the  
meta-model to make the final loan approval decision.
""")

# ==================================================
# PREDICTION
# ==================================================
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    input_data = np.array([[ 
        app_income,
        co_income,
        loan_amount,
        loan_term,
        credit_history,
        self_employed,
        property_area
    ]])

    input_scaled = scaler.transform(input_data)

    # Base model predictions
    lr_pred = lr.predict(input_scaled)[0]
    dt_pred = dt.predict(input_scaled)[0]
    rf_pred = rf.predict(input_scaled)[0]

    # Final stacking prediction
    final_prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][final_prediction] * 100

    # ==================================================
    # OUTPUT
    # ==================================================
    st.subheader("📌 Loan Eligibility Result")

    if final_prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📊 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred else 'Rejected'}")

    st.markdown("### 🧠 Final Stacking Decision")
    st.write("**Approved**" if final_prediction == 1 else "**Rejected**")

    st.markdown("### 📈 Confidence Score")
    st.write(f"{confidence:.2f}%")

    # ==================================================
    # BUSINESS EXPLANATION (MANDATORY)
    # ==================================================
    st.subheader("💼 Business Explanation")

    st.info(
        f"Based on the applicant’s income, credit history, loan amount, "
        f"and combined predictions from multiple machine learning models, "
        f"the applicant is {'likely' if final_prediction == 1 else 'unlikely'} "
        f"to repay the loan. Therefore, the stacking model predicts loan "
        f"{'approval' if final_prediction == 1 else 'rejection'} "
        f"with a confidence of {confidence:.2f}%."
    )
