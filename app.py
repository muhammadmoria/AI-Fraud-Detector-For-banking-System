import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Fraud Detection System for Banking",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the saved model
model = joblib.load('fraud_detection_model.pkl')

# Load the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('creditcard.csv')
        if data.empty:
            st.error("The dataset is empty. Please check the file content.")
        return data
    except FileNotFoundError:
        st.error("The dataset file 'creditcard.csv' was not found.")
    except Exception as e:
        st.error(f"An error occurred while loading the data: {str(e)}")
    return None

# Initialize dataset
data = load_data()

# Page Layout
st.markdown("""<style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 1.8em;
        color: #3498DB;
        font-weight: bold;
        margin-top: 30px;
        text-align: left;
    }
    .section-content {
        text-align: center;
    }
</style>""", unsafe_allow_html=True)

st.markdown("""# 📊 Welcome to the AI-Powered Fraud Detection System for Banking""")
st.markdown("""Your one-stop solution for fraud prevention! 🚀""")

tab1, tab2, tab3 = st.tabs(["🏠Home", "📋Predict Fraud-Detection", "✍️ Provide Feedback"])

# Page 1: Home
with tab1:
    st.markdown("""
        ## 👋 About Me  
        Hi! I'm Muhammad Dawood, a data scientist specializing in machine learning, deep learning, and Natural Language Processing (NLP).  
        I’m passionate about building intelligent systems and data-driven applications, particularly those enhancing security and user experience in financial services.

        ## 🔍 Project Overview  
        This project focuses on identifying fraudulent transactions in banking data to help financial institutions prevent and mitigate fraud in real-time. Here’s what we’ve accomplished:
        - **Data Collection** 📊: Analyzed a large dataset of banking transactions from Kaggle, using feature engineering to capture critical transaction patterns and behaviors.
        - **Advanced Fraud Detection** 🔍: Leveraged sophisticated machine learning algorithms to detect anomalies and suspicious activities, predicting potential fraud effectively.
        - **Model Optimization** 📈: Tuned model parameters to maximize accuracy, ensuring robust performance for detecting fraudulent behavior across diverse transactions.
        - **Deployment** 🌐: Developed an intuitive app interface using Streamlit, enabling secure, real-time fraud analysis and detection.

        ## 💻 Technologies Used  
        - **Languages & Libraries**: Python, Pandas, Scikit-Learn, Joblib for model persistence, and Streamlit for user interaction.
        - **Deployment**: Streamlit app for user-friendly, real-time fraud detection.
    """)

# Page 2: Predict Fraud Detection
# Page 2: Predict Fraud Detection
with tab2:
    st.title("🔍 Predict Fraud Detection")
    st.write("Fill in the transaction details below to check for fraud.")

    # Define input fields with three-column layout
    def get_user_input():
        col1, col2, col3 = st.columns(3)

        with col1:
            Time = st.number_input("Time", value=452.0)
            V1 = st.number_input("V1", value=-1.359807)
            V2 = st.number_input("V2", value=-0.072781)
            V3 = st.number_input("V3", value=2.536346)
            V4 = st.number_input("V4", value=1.378155)
            V5 = st.number_input("V5", value=-0.338321)
            V6 = st.number_input("V6", value=0.462388)
            V7 = st.number_input("V7", value=0.239599)
            V8 = st.number_input("V8", value=0.098698)

        with col2:
            V9 = st.number_input("V9", value=0.363787)
            V10 = st.number_input("V10", value=0.090794)
            V11 = st.number_input("V11", value=-0.5516)
            V12 = st.number_input("V12", value=-0.6178)
            V13 = st.number_input("V13", value=-0.991)
            V14 = st.number_input("V14", value=-0.311169)
            V15 = st.number_input("V15", value=1.468177)
            V16 = st.number_input("V16", value=-0.4704)
            V17 = st.number_input("V17", value=0.20797)

        with col3:
            V18 = st.number_input("V18", value=0.0258)
            V19 = st.number_input("V19", value=0.403992)
            V20 = st.number_input("V20", value=0.251412)
            V21 = st.number_input("V21", value=-0.018307)
            V22 = st.number_input("V22", value=0.277838)
            V23 = st.number_input("V23", value=-0.110473)
            V24 = st.number_input("V24", value=0.066928)
            V25 = st.number_input("V25", value=0.128539)
            V26 = st.number_input("V26", value=-0.189114)
        V27 = st.number_input("V27", value=0.133558)
        V28 = st.number_input("V28", value=-0.021053)
        Amount = st.number_input("Amount", value=149.62)

        user_data = pd.DataFrame({
            "Time": [Time], "V1": [V1], "V2": [V2], "V3": [V3], "V4": [V4], "V5": [V5],
            "V6": [V6], "V7": [V7], "V8": [V8], "V9": [V9], "V10": [V10], "V11": [V11],
            "V12": [V12], "V13": [V13], "V14": [V14], "V15": [V15], "V16": [V16], "V17": [V17],
            "V18": [V18], "V19": [V19], "V20": [V20], "V21": [V21], "V22": [V22], "V23": [V23],
            "V24": [V24], "V25": [V25], "V26": [V26], "V27": [V27], "V28": [V28], "Amount": [Amount]
        })

        return user_data

    # Capture user input
    input_data = get_user_input()

    # Make prediction and display results
    if st.button("Predict Fraud"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"Prediction: Fraud detected with a probability of {prediction_proba:.2f}")
        else:
            st.success(f"Prediction: No fraud detected with a probability of {1 - prediction_proba:.2f}")

    # Display a random sample from the dataset with a refresh button
    st.subheader("Sample Data")
    if data is not None and not data.empty:
        if st.button("Refresh Sample"):
            sample_data = data.sample(n=5)  # Display 5 random rows from the dataset
        else:
            sample_data = data.sample(n=5)

        st.write("Here is a random sample from the dataset:")
        st.dataframe(sample_data)
    else:
        st.error("No data available to sample. Please check the dataset.")

# Page 3: Provide Feedback
with tab3:
    st.title("💬 Provide Feedback")
    st.write("We value your feedback! Please share your experience with this app.")

    # Feedback form
    name = st.text_input("Name", "")
    feedback = st.text_area("Feedback", "")
    submit_feedback = st.button("Submit Feedback")

    if submit_feedback and name and feedback:
        with open("feedback.txt", "a") as f:
            f.write(f"{datetime.now().date()} - {name}: {feedback}\n")
        st.success("Thank you for your feedback!")
    elif submit_feedback:
        st.warning("Please enter both your name and feedback.")

    # Display previous feedback
    st.subheader("Previous Feedback")
    try:
        with open("feedback.txt", "r") as f:
            feedback_history = f.readlines()
            for line in feedback_history:
                st.write(line)
    except FileNotFoundError:
        st.info("No feedback has been provided yet.")
