"""Simple UI:  streamlit run app.py"""
import os
import pandas as pd
import streamlit as st
from data_generator import generate
from features import LOCATIONS, FEATURES, to_matrix
from train import run

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor")
st.write("A **Linear Regression** model trained with **Gradient Descent** (scikit-learn) "
         "on 12,000 houses.")

@st.cache_data
def load():
    if not os.path.exists("house_data.csv"):
        generate().to_csv("house_data.csv", index=False)
    return pd.read_csv("house_data.csv")

@st.cache_resource
def train_model(lr, epochs):
    return run(load(), lr=lr, epochs=epochs)

df = load()
with st.sidebar.expander("⚙️ Training settings (optional)"):
    lr = st.select_slider("Learning rate", [0.001, 0.005, 0.01, 0.05, 0.1], 0.01)
    epochs = st.slider("Epochs (rounds of learning)", 10, 300, 100, 10)
with st.spinner("Training the model..."):
    res = train_model(lr, epochs)
model = res["model"]
coef, intercept = model.coefficients()

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Price", "📋 All Data",
                                  "📉 Training & MSE", "⚖️ What the model learned"])

# ---------- 1. Predict ----------
with tab1:
    st.subheader("Enter the house details")
    c1, c2, c3 = st.columns(3)
    with c1:
        loc = st.selectbox("Location", list(LOCATIONS), 5)
        st.caption(LOCATIONS[loc][1])
        area = st.number_input("Area (sq ft)", 350, 6000, 1800, 50)
        bed = st.slider("Bedrooms", 1, 6, 3)
        bath = st.slider("Bathrooms", 1, 6, 2)
        floors = st.selectbox("Floors", [1, 2, 3], 1)
    with c2:
        age = st.slider("House age (years)", 0, 60, 10)
        garage = st.slider("Garage spaces", 0, 3, 1)
        cond = st.slider("Condition (1 = poor, 10 = excellent)", 1, 10, 7)
        furn = st.selectbox("Furnishing", [0, 1, 2],
                            format_func=lambda i: ["Unfurnished", "Semi-furnished", "Fully furnished"][i])
    with c3:
        dc = st.number_input("Distance to city centre (km)", 0.3, 50.0, 5.0, 0.5)
        ds = st.number_input("Distance to school (km)", 0.2, 15.0, 2.0, 0.2)
        crime = st.slider("Crime index (0 = safe, 100 = unsafe)", 1, 100, 35)
        pool = st.checkbox("Swimming pool")
        garden = st.checkbox("Garden", True)
    row = pd.DataFrame([dict(location=loc, area_sqft=area, bedrooms=bed, bathrooms=bath,
        floors=floors, age_years=age, garage_spaces=garage, dist_center_km=dc,
        dist_school_km=ds, crime_index=crime, condition_score=cond,
        has_pool=int(pool), has_garden=int(garden), furnishing=furn)])
    x = to_matrix(row)
    price = float(model.predict(x)[0])
    st.divider()
    st.metric("💰 Estimated price", f"$ {price:,.0f}")
    st.markdown("**How each feature changed the price ($):**")
    contrib = pd.Series(x[0] * coef, index=FEATURES)
    st.bar_chart(contrib[contrib != 0].sort_values())
    st.caption(f"Starting value (intercept): ${intercept:,.0f}. "
               "Starting value + all bars = estimated price.")

# ---------- 2. All data ----------
with tab2:
    st.subheader(f"Full dataset: {len(df):,} rows × {df.shape[1]} columns")
    st.caption("Scroll inside the table to see every row. Click a column name to sort.")
    st.dataframe(df, height=550, width="stretch")
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "house_data.csv")

# ---------- 3. Training & MSE ----------
with tab3:
    st.subheader("Model error (MSE)")
    a, b = st.columns(2)
    a.metric("Training MSE", f"{res['train']['MSE']:,.0f}")
    b.metric("Test MSE", f"{res['test']['MSE']:,.0f}")
    st.info("**MSE (Mean Squared Error)** = average of (actual price − predicted price)². "
            "Lower is better. If Training MSE and Test MSE are close, the model works "
            "well on new houses too.")
    st.subheader("Learning curve")
    h = model.history
    st.line_chart(pd.DataFrame({"Training loss": h["train_loss"],
                                "Test loss": h["val_loss"]}))
    st.caption("The line goes down and flattens: the model is learning, "
               "then reaches its best weights.")
    st.subheader("Sample predictions (unseen test houses)")
    sample = res["te"].head(15)[["location", "area_sqft", "bedrooms", "price"]].copy()
    sample.columns = ["Location", "Area", "Bedrooms", "Actual price"]
    sample["Predicted price"] = res["y_pred"][:15].round(-2)
    st.dataframe(sample, width="stretch")

# ---------- 4. Weights ----------
with tab4:
    st.subheader("Learned weights")
    st.write("Each number shows how much the price changes when that feature "
             "increases by 1 (or, for locations, compared with **Rural**).")
    w = pd.DataFrame({"Price change ($) per unit": coef.round(2)}, index=FEATURES)
    st.dataframe(w, width="stretch", height=400)
    st.bar_chart(w)
