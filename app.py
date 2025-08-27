import streamlit as st
import pandas as pd
import joblib  # For loading the model

# Load trained spam detection model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Email Spam Detector - Shailesh,Shakti,Sharique and Shivam", page_icon="📧", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #4B8BBE;
            text-align: center;
        }
        .subtitle {
            color: #888;
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 30px;
        }
        .stTextArea label {
            font-size: 1.1em;
            font-weight: 500;
        }
        .stButton > button {
            background-color: #4B8BBE;
            color: white;
            font-weight: bold;
        }
        .result-box {
            font-size: 1.3em;
            font-weight: bold;
            background-color: #gray;
            border-left: 5px solid #34a853;
            padding: 10px;
            margin-top: 20px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title">📧 Email Spam Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Built with ❤️ by <b>Shailesh, Shakti, Sharique and Shivam</b></div>', unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["📩 Single Message", "📂 Bulk CSV File"])

# ------------------------ SINGLE EMAIL DETECTION ------------------------
with tab1:
    st.markdown("### 🔍 Check a Single Email")

    user_input = st.text_area("✍️ Enter an email message:", height=150, placeholder="Type your email content here...")

    if st.button("🚀 Detect Spam"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            result = "🚨 Spam!" if prediction == 1 else "✅ Not Spam (Ham)"

            st.markdown(f"<div class='result-box'>Prediction: {result}</div>", unsafe_allow_html=True)

# ------------------------ BULK CSV FILE DETECTION ------------------------
with tab2:
    st.markdown("### 📂 Upload a CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file (must contain a column named 'text')", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.markdown("#### 📊 CSV File Preview:")
        st.dataframe(uploaded_df.head())

        if "text" not in uploaded_df.columns:
            st.error("❌ CSV file must contain a 'text' column with email content.")
        else:
            X_test = vectorizer.transform(uploaded_df["text"])
            predictions = model.predict(X_test)
            uploaded_df["Prediction"] = ["🚨 Spam" if label == 1 else "✅ Not Spam" for label in predictions]

            st.markdown("#### ✅ Prediction Results:")
            st.dataframe(uploaded_df[["text", "Prediction"]])

            @st.cache_data
            def convert_df(dataframe):
                return dataframe.to_csv(index=False).encode('utf-8')

            csv = convert_df(uploaded_df)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="spam_predictions.csv",
                mime="text/csv",
            )

# --- FOOTER ---
st.markdown("""
    <hr style="margin-top: 40px;">
    <center>
        <span style="font-size: 0.9em; color: #888;">Made with Streamlit | © 2025 Shailesh Pandit</span>
    </center>
""", unsafe_allow_html=True)
