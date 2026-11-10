import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide"
)

# =========================
# LOAD MODEL FILES
# =========================

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
oe = joblib.load("ordinal_encoder.pkl")

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
}

.main-title {
    text-align: center;
    color: #38bdf8;
    font-size: 55px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 20px;
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg,#06b6d4,#3b82f6);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
}

.footer {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    border: none;
    font-size: 20px;
    font-weight: bold;
    color: white;
    background: linear-gradient(90deg,#8b5cf6,#3b82f6);
}

.stButton > button:hover {
    background: linear-gradient(90deg,#7c3aed,#2563eb);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown(
    "<h1 class='main-title'>💎 Diamond Price Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning Based Diamond Valuation System</div>",
    unsafe_allow_html=True
)

st.image(
    "https://images.unsplash.com/photo-1615655114867-d3f0f8f6f1f9?w=1200",
    use_container_width=True
)

st.write("")

# =========================
# METRICS
# =========================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("💎 Dataset", "54,000+")

with c2:
    st.metric("🤖 Algorithm", "KNN")

with c3:
    st.metric("📊 Features", "5")

st.divider()

# =========================
# INPUT SECTION
# =========================

st.subheader("Enter Diamond Details")

col1, col2 = st.columns(2)

with col1:

    carat = st.number_input(
        "💠 Carat",
        min_value=0.1,
        value=0.5,
        step=0.1
    )

    depth = st.number_input(
        "📏 Depth",
        value=61.5
    )

with col2:

    table = st.number_input(
        "📈 Table",
        value=57.0
    )

    y = st.number_input(
        "📐 Width (Y)",
        value=4.25
    )

clarity = st.selectbox(
    "✨ Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)

st.write("")

# =========================
# PREDICTION
# =========================

if st.button("🔮 Predict Diamond Price"):

    input_data = pd.DataFrame({
        "carat": [carat],
        "depth": [depth],
        "table": [table],
        "y": [y]
    })

    input_scaled = scaler.transform(input_data)

    input_encoded = oe.transform([[clarity]])

    input_variable = pd.DataFrame({
        "carat": [input_scaled[0, 0]],
        "clarity": [input_encoded[0, 0]],
        "depth": [input_scaled[0, 1]],
        "table": [input_scaled[0, 2]],
        "y": [input_scaled[0, 3]]
    })

    prediction = model.predict(input_variable)

    price = prediction[0]

    if price < 5000:
        category = "🟢 Budget Diamond"
    elif price < 15000:
        category = "🟡 Premium Diamond"
    else:
        category = "🔴 Luxury Diamond"

    st.markdown(
        f"""
        <div class="result-card">
            <h2>💰 Estimated Diamond Price</h2>
            <h1>${price:,.2f}</h1>
            <h3>{category}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()

# =========================
# PROJECT DETAILS
# =========================

st.divider()

with st.expander("📖 About This Project"):

    st.markdown("""
### Features
- Carat
- Clarity
- Depth
- Table
- Width (Y)

### Machine Learning Techniques
- KNN Regression
- Ordinal Encoding
- Standard Scaling

### Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-Learn
- Joblib
""")

# =========================
# FOOTER
# =========================

st.write("")
st.divider()

st.markdown("""
<div class='footer'>
<h2>💎 Diamond Price Prediction System</h2>
<p>Built with Streamlit & Machine Learning</p>
<p>🚀 Developed by Vivekananda Reddy</p>
</div>
""", unsafe_allow_html=True)
