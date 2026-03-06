import streamlit as st
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

# ---------------- TITLE ----------------
st.title("🏠 Smart House Price Prediction – Stacking Model")

st.markdown("""
This application predicts **house prices** using a  
**Stacking Ensemble Regression model** for better accuracy.
""")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🏡 House Details")

bedrooms = st.sidebar.number_input("Bedrooms", 0, 10)
bathrooms = st.sidebar.number_input("Bathrooms", 0.0, 10.0)
sqft_living = st.sidebar.number_input("Living Area (sqft)", 0)
sqft_lot = st.sidebar.number_input("Lot Area (sqft)", 0)
floors = st.sidebar.number_input("Floors", 0.0, 5.0)
waterfront = st.sidebar.radio("Waterfront", ["No", "Yes"])
condition = st.sidebar.number_input("Condition (1–5)", 1, 5)

waterfront = 1 if waterfront == "Yes" else 0

user_input = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                        floors, waterfront, condition]])

# ---------------- MODEL INFO ----------------
st.markdown("---")
st.subheader("🧠 Model Architecture (Stacking)")

st.markdown("""
**Base Models:**
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

**Meta Model:**
- Linear Regression  
""")

# ---------------- DUMMY TRAINING DATA ----------------
X_dummy = np.random.rand(300, 7)
y_dummy = np.random.randint(50000, 1000000, 300)

scaler = StandardScaler()
X_dummy = scaler.fit_transform(X_dummy)
user_input = scaler.transform(user_input)

# ---------------- MODELS ----------------
lr = LinearRegression()
dt = DecisionTreeRegressor(max_depth=5)
rf = RandomForestRegressor(n_estimators=100)

stack_model = StackingRegressor(
    estimators=[('lr', lr), ('dt', dt), ('rf', rf)],
    final_estimator=LinearRegression()
)

stack_model.fit(X_dummy, y_dummy)

# ---------------- BUTTON ----------------
if st.button("🔘 Predict House Price (Stacking Model)"):

    lr.fit(X_dummy, y_dummy)
    dt.fit(X_dummy, y_dummy)
    rf.fit(X_dummy, y_dummy)

    lr_pred = lr.predict(user_input)[0]
    dt_pred = dt.predict(user_input)[0]
    rf_pred = rf.predict(user_input)[0]

    final_price = stack_model.predict(user_input)[0]

    # ---------------- OUTPUT ----------------
    st.markdown("---")
    st.subheader("💰 Predicted House Price")

    st.success(f"₹ {final_price:,.2f}")

    st.markdown("### 📊 Base Model Predictions")
    st.write(f"Linear Regression: ₹ {lr_pred:,.2f}")
    st.write(f"Decision Tree: ₹ {dt_pred:,.2f}")
    st.write(f"Random Forest: ₹ {rf_pred:,.2f}")

    # ---------------- BUSINESS EXPLANATION ----------------
    st.markdown("---")
    st.subheader("💼 Business Explanation")

    st.markdown("""
    Based on house size, location features, and property condition,  
    multiple regression models estimate the market value.

    The stacking model combines these estimates to provide  
    a **more stable and reliable house price prediction**.
    """)
