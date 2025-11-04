# ------------------------------------------------------------
#  Streamlit Web App - EMIPredict AI
# ------------------------------------------------------------

# Import necessary libraries
import streamlit as st          # Streamlit for building the web UI
import pandas as pd             # For handling data
import numpy as np              # For numeric calculations
import joblib                   # For loading the trained models
import matplotlib.pyplot as plt # For creating charts

# ------------------------------------------------------------
# 1 Load the saved models
# ------------------------------------------------------------
# We use models saved in Step 4
class_model = joblib.load("best_classification_model.pkl")   # For EMI eligibility prediction
reg_model = joblib.load("best_regression_model.pkl")         # For max EMI prediction

# ------------------------------------------------------------
# 2 Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="EMIPredict AI - Intelligent Financial Risk Assessment Platform",
    page_icon=" ",
    layout="centered"
)

# Title and description
st.title(" EMIPredict AI")
st.markdown("""
### Intelligent Financial Risk Assessment Platform
This platform uses Machine Learning to predict:
- **EMI Eligibility (Eligible / High Risk / Not Eligible)**
- **Maximum Safe Monthly EMI Amount**

---
""")

# ------------------------------------------------------------
# 3 Collect user input
# ------------------------------------------------------------

st.header(" Enter Your Financial Details")

# Group inputs in columns for cleaner UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=40, value=5)
    company_type = st.selectbox("Company Type", ["Small", "Medium", "Large"])

with col2:
    monthly_salary = st.number_input("Monthly Salary (₹)", min_value=10000, max_value=500000, value=50000)
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, max_value=100000, value=10000)
    family_size = st.number_input("Family Size", min_value=1, max_value=10, value=4)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, value=2)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    existing_loans = st.selectbox("Existing Loans?", ["Yes", "No"])

# ------------------------------------------------------------
# 4 Additional monthly expenses
# ------------------------------------------------------------
st.subheader(" Monthly Expenses")

col3, col4 = st.columns(2)
with col3:
    school_fees = st.number_input("School Fees (₹)", min_value=0, max_value=100000, value=5000)
    college_fees = st.number_input("College Fees (₹)", min_value=0, max_value=100000, value=3000)
    travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, max_value=50000, value=2000)
with col4:
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0, max_value=100000, value=10000)
    other_monthly_expenses = st.number_input("Other Expenses (₹)", min_value=0, max_value=100000, value=2000)
    bank_balance = st.number_input("Bank Balance (₹)", min_value=0, max_value=1000000, value=50000)
    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, max_value=1000000, value=10000)

# EMI scenario
emi_scenario = st.selectbox(
    "EMI Scenario",
    ["E-commerce Shopping", "Home Appliances", "Vehicle", "Personal Loan", "Education"]
)

requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000, max_value=2000000, value=200000)
requested_tenure = st.number_input("Requested Tenure (Months)", min_value=3, max_value=84, value=24)

# ------------------------------------------------------------
#  Create a DataFrame for prediction
# ------------------------------------------------------------
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "marital_status": [marital_status],
    "education": [education],
    "monthly_salary": [monthly_salary],
    "employment_type": [employment_type],
    "years_of_employment": [years_of_employment],
    "company_type": [company_type],
    "house_type": [house_type],
    "monthly_rent": [monthly_rent],
    "family_size": [family_size],
    "dependents": [dependents],
    "school_fees": [school_fees],
    "college_fees": [college_fees],
    "travel_expenses": [travel_expenses],
    "groceries_utilities": [groceries_utilities],
    "other_monthly_expenses": [other_monthly_expenses],
    "existing_loans": [existing_loans],
    "credit_score": [credit_score],
    "bank_balance": [bank_balance],
    "emergency_fund": [emergency_fund],
    "emi_scenario": [emi_scenario],
    "requested_amount": [requested_amount],
    "requested_tenure": [requested_tenure]
})

# ------------------------------------------------------------
#  Predict Results (When Button Clicked)
# ------------------------------------------------------------
if st.button(" Predict EMI Eligibility & Max EMI"):
    # Label encode
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = input_data[col].astype('category').cat.codes

    # Derived features
    input_data["current_emi_amount"] = input_data["requested_amount"] / input_data["requested_tenure"]
    input_data["total_expenses"] = (
        input_data["school_fees"] + input_data["college_fees"] +
        input_data["travel_expenses"] + input_data["groceries_utilities"] +
        input_data["other_monthly_expenses"] + input_data["monthly_rent"]
    )
    input_data["debt_to_income_ratio"] = input_data["total_expenses"] / (input_data["monthly_salary"] + 1)
    input_data["expense_to_income_ratio"] = input_data["total_expenses"] / (input_data["monthly_salary"] + 1)
    input_data["affordability_ratio"] = input_data["requested_amount"] / (input_data["monthly_salary"] * input_data["requested_tenure"])
    input_data["credit_utilization_ratio"] = (850 - input_data["credit_score"]) / 850
    input_data["employment_stability"] = input_data["years_of_employment"] / 40

    #  Reorder features to match model training
    expected_features = [
        'age', 'gender', 'marital_status', 'education', 'monthly_salary',
        'employment_type', 'years_of_employment', 'company_type', 'house_type',
        'monthly_rent', 'family_size', 'dependents', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
        'emergency_fund', 'emi_scenario', 'requested_amount', 'requested_tenure',
        'debt_to_income_ratio', 'total_expenses', 'expense_to_income_ratio',
        'affordability_ratio', 'credit_utilization_ratio', 'employment_stability'
    ]
    input_data = input_data[expected_features]

    # Predictions
    class_pred = class_model.predict(input_data)[0]
    reg_pred = reg_model.predict(input_data)[0]

    # Results
    st.subheader(" Prediction Results")
    st.write(f"**Predicted EMI Eligibility:** `{class_pred}`")
    st.write(f"**Maximum Safe EMI Amount:** ₹{reg_pred:,.2f}")
    # --------------------------------------------------------
    # 7 Visualization: Pie Chart of EMI Breakdown
    # --------------------------------------------------------
    principal = requested_amount / requested_tenure
    interest = reg_pred * 0.15  # Rough 15% interest assumption for visualization
    plt.figure(figsize=(4, 4))
    plt.pie(
        [principal, interest],
        labels=["Principal", "Interest"],
        autopct='%1.1f%%',
        colors=['#66b3ff', '#ff9999']
    )
    plt.title("Estimated EMI Composition")
    st.pyplot(plt)

    # Success message
    st.success(" Prediction completed successfully!")

# Footer
st.markdown("---")
st.caption("Developed by Aakash • Powered by Streamlit & MLflow • © 2025 EMIPredict AI")
