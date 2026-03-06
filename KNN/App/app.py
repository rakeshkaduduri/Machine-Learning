import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ==================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==================================================
st.set_page_config(
    page_title="Customer Risk Prediction System",
    layout="wide"
)

# ==================================================
# LOAD CSS
# ==================================================
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==================================================
# LOAD DATASET
# ==================================================
df = pd.read_csv("../KNN/credit_risk_dataset.csv")

# ==================================================
# FEATURE SELECTION
# ==================================================
features = [
    'person_age',
    'person_income',
    'loan_amnt',
    'cb_person_cred_hist_length'
]

X = df[features]
y = df['loan_status']   # 1 = Low Risk, 0 = High Risk

# ==================================================
# SCALING & TRAIN-TEST SPLIT
# ==================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==================================================
# APP HEADER
# ==================================================
st.title("Customer Risk Prediction System (KNN)")
st.write(
    "This system predicts customer risk by comparing them with similar customers."
)

st.markdown("---")

# ==================================================
# SIDEBAR INPUTS
# ==================================================
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)

income = st.sidebar.number_input(
    "Annual Income",
    min_value=0,
    step=1000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    step=500
)

credit_history_years = st.sidebar.slider(
    "Credit History Length (Years)",
    0,
    30,
    5
)

k_value = st.sidebar.slider(
    "K Value (Number of Neighbors)",
    1,
    15,
    5
)

# ==================================================
# PREDICTION LOGIC
# ==================================================
if st.button("Predict Customer Risk"):

    # -----------------------------
    # INPUT VALIDATION
    # -----------------------------
    if loan_amount <= 0:
        st.error("❌ Loan amount must be greater than 0 to proceed.")

    # -----------------------------
    # INCOME-BASED BUSINESS RULE
    # -----------------------------
    elif income < 10000:
        st.markdown("## Prediction Result")
        st.markdown(
            "<h2 style='color:red;'>🔴 High Risk Customer</h2>",
            unsafe_allow_html=True
        )

        st.warning(
            "Income is below the minimum eligibility threshold."
        )

        st.markdown("## Business Insight")
        st.info(
            "Customers with very low income are classified as High Risk "
            "due to insufficient repayment capacity. "
            "This decision is made before applying machine learning."
        )

    # -----------------------------
    # KNN SIMILARITY-BASED PREDICTION
    # -----------------------------
    else:
        user_data = np.array([
            [age, income, loan_amount, credit_history_years]
        ])

        user_scaled = scaler.transform(user_data)

        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_train, y_train)

        prediction = knn.predict(user_scaled)[0]

        distances, indices = knn.kneighbors(user_scaled)
        neighbor_labels = y_train.iloc[indices[0]]

        # -----------------------------
        # PREDICTION OUTPUT
        # -----------------------------
        st.markdown("## Prediction Result")

        if prediction == 0:
            st.markdown(
                "<h2 style='color:red;'>🔴 High Risk Customer</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color:green;'>🟢 Low Risk Customer</h2>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # -----------------------------
        # NEAREST NEIGHBORS EXPLANATION
        # -----------------------------
        st.markdown("## Nearest Neighbors Explanation")

        st.write(f"**Number of neighbors considered:** {k_value}")

        st.write(
            f"**Majority class among neighbors:** "
            f"{'Low Risk' if neighbor_labels.mean() > 0.5 else 'High Risk'}"
        )

        neighbor_data = df.iloc[indices[0]][features].copy()
        neighbor_data['Risk Label'] = neighbor_labels.values
        neighbor_data['Risk Label'] = neighbor_data['Risk Label'].map(
            {0: "High Risk", 1: "Low Risk"}
        )

        st.dataframe(neighbor_data)

        # -----------------------------
        # BUSINESS INSIGHT
        # -----------------------------
        st.markdown("## Business Insight")

        st.info(
            "This decision is based on the applicant’s income and similarity "
            "with nearby customers in feature space. "
            "KNN evaluates historical repayment behavior of similar profiles "
            "to assign customer risk."
        )
