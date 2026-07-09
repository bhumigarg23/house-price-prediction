import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("house_price_model.pkl")

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏠 House Price Prediction using Machine Learning")
st.markdown(
    "Predict the selling price of a house using a **Gradient Boosting Regressor**."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📌 Project Information")

st.sidebar.success("Model: Gradient Boosting Regressor")

st.sidebar.metric(
    label="R² Score",
    value="0.881"
)

st.sidebar.markdown("---")

st.sidebar.write("""
### Features Used

- Overall Quality
- Living Area
- Garage Cars
- Garage Area
- Basement Area
- Year Built
- House Age
- Full Bathrooms
""")

# -----------------------------
# Input Section
# -----------------------------
st.header("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality", 1, 10, 5)

    gr_liv_area = st.number_input(
        "Ground Living Area (sq ft)",
        min_value=300,
        max_value=6000,
        value=1500
    )

    garage_cars = st.slider(
        "Garage Cars",
        0,
        5,
        2
    )

    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=1500,
        value=500
    )

with col2:
    total_bsmt = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=3000,
        value=800
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2025,
        value=2000
    )

    house_age = st.number_input(
        "House Age",
        min_value=0,
        max_value=150,
        value=25
    )

    full_bath = st.slider(
        "Full Bathrooms",
        0,
        5,
        2
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict House Price"):

    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "GarageArea": [garage_area],
        "TotalBsmtSF": [total_bsmt],
        "YearBuilt": [year_built],
        "HouseAge": [house_age],
        "FullBath": [full_bath]
    })

    prediction = model.predict(input_data)

    st.markdown("## 💰 Estimated House Price")

    st.metric(
        label="Predicted Price",
        value=f"${prediction[0]:,.0f}"
    )

# -----------------------------
# Feature Importance
# -----------------------------
st.markdown("---")

st.subheader("📊 Feature Importance")

importance = pd.Series(
    model.feature_importances_,
    index=[
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "YearBuilt",
        "HouseAge",
        "FullBath"
    ]
)

fig, ax = plt.subplots(figsize=(8,4))


importance.sort_values(ascending=True).plot(kind="barh", ax=ax)

ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")

st.pyplot(fig)
plt.close(fig)
# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "Developed by Bhumi | House Price Prediction using Machine Learning"
)