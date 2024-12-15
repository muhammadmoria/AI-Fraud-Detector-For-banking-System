import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import random

st.set_page_config(
    page_title="DefrauderAI ",
    page_icon="🧠⚠️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the saved model
model = joblib.load('fraud_detection_model.pkl')

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

# Initialize dataset
data = load_data()

st.markdown("""
    <style>
        /* Background and General Layout */
        body {
            background: linear-gradient(135deg, #141E30, #243B55);
            color: #E0E0E0;
            font-family: 'Roboto', sans-serif;
        }

        /* Main Title with Sticker */
        .main-title {
            font-size: 2.8em;
            font-weight: bold;
            color: #FFD700;
            text-align: center;
            margin-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
        }

        /* Section Title Styling */
        .section-title {
            font-size: 1.8em;
            color: #FFD700;
            font-weight: bold;
            margin-top: 15px;
            text-align: center;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
        }

        /* Card Styling */
        .card {
            background: #7393B3;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: scale(1.02);
        }

        /* Feedback and Input Fields */
        .feedback-box {
            background: #2F2F3B;
            color: #E0E0E0;
            padding: 5px;
            border-radius: 8px;
            border: 1px solid #FFD700;
            margin-bottom: 10px;
        }

        /* Input Field Styling */
        .stTextInput > div, .stTextArea > div {
            background-color: #333945;
            color: #FFFFFF;
            border-radius: 5px;
            border: 1px solid #FFD700;
            padding: 5px;
        }

        /* Responsive Buttons with Stickers */
        .stButton>button {
            background-color: #6082B6;
            color: #FFFF00;
            font-size: 1.1em;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            border: none;
            position: relative;
            padding-left: 35px;
        }
        .stButton>button::before {
            content: "✨";
            position: absolute;
            left: 10px;
            top: 3px;
            font-size: 1.2em;
        }
        .stButton>button:hover {
            background-color: #6082B6;
            color: #FFFF00;
        }

        /* Uniform Recommendation Boxes */
        .recommend-box {
            background: #333945;
            color: #FFD700;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin: 5px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        }
        .recommend-box img {
            border-radius: 5px;
            height: 180px;
            width: 100%;
            object-fit: contain;
        }

        /* Profile Buttons */
        .profile-buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 15px;
        }

        .profile-buttons a {
            text-decoration: none;
            font-weight: bold;
            color: white;
            background: #6082B6;
            padding: 10px 20px;
            border-radius: 8px;
            transition: background-color 0.3s;
        }

        .profile-buttons a:hover {
            background: #FFC300;
            color: #2C3E50;
        }

        @media (max-width: 768px) {
            .main-title { font-size: 2.2em; }
            .section-title { font-size: 1.6em; }
            .profile-buttons {
                flex-direction: column;
                gap: 8px;
            }

            .profile-buttons a {
                text-align: center;
                padding: 12px;
            }
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""<div class="main-title"> DefrauderAI 🧠⚠️</div>""", unsafe_allow_html=True)
st.markdown("""<div class="main-title"> 🚨 AI-Powered Fraud Detection System for Banking </div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📋 Recommender", "💬 Feedback"])

# Page 1: Home
with tab1:
    st.markdown("""
            <div class="card">
                <h3 class="section-title">👤 About Me</h3>
                <p>
                    Hi! I'm Muhammad Dawood, a data scientist specializing in machine learning, deep learning, and NLP.
                    My passion lies in building intelligent systems and web applications that enhance user experiences.
                </p>
            </div>

            <div class="card">
                <h3 class="section-title">🚀 Project Overview</h3>
                <p>
            This AI-powered fraud detection system named as DefrauderAI uses machine learning to identify fraudulent banking transactions in real-time, helping financial institutions prevent fraud effectively.
                </p>
            </div>
                    <div class="card">
                <h3 class="section-title">⚙️ Technologies Used</h3>
                <ul style="list-style: none; padding-left: 10px;">
                    <li>🐍 <b>Python</b> - Programming Language</li>
                    <li>🤖 <b>Machine Learning</b> - Recommendation Engine</li>
                    <li>📊 <b>Data Science</b> - Data Processing and Analysis</li>
                    <li>🧠 <b>Deep Learning</b> - Model Improvement</li>
                    <li>💡 <b>NLP</b> - Natural Language Processing for Recommendations</li>
                </ul>
            </div>

            <div class="profile-buttons">
                <a href="https://github.com/muhammadmoria" target="_blank">GitHub</a>
                <a href="https://www.linkedin.com/in/muhammaddawood361510306/" target="_blank">LinkedIn</a>
                <a href="https://muhammadmoria.github.io/portfolio-new/">Portfolio</a>
                <a href="https://wa.me/923709152202" target="_blank">WhatsApp</a>
            
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer">
        <center>
            <p>Gmail : <b>muhammaddawoodmoria@gmail.com</b></p>
            </center>
        </div>
    """, unsafe_allow_html=True)

# Page 2: Predict Fraud Detection
with tab2:
    st.markdown("""<div class="card"> <h1 style="text-align: center;">🔍 Predict Fraud Detection</h1> </div>""", unsafe_allow_html=True)
    st.write("Fill in the transaction details below to check for fraud.")
    
    st.markdown("""<div class="card">
                    <p>      
                        <h3 style="text-align: center;">📝 What to Fill?</h3> 
                        1.Time ⏱️: Time since the first transaction.
                        <br>
                        2.Amount 💰: Transaction amount in dollars.  
                        <br>
                        3.V1 to V28 🔍: Behavior scores from transaction data (use provided or known values).
                    </p>
                    </div>""", unsafe_allow_html=True)
    

    # Define input fields with three-column layout
    def get_user_input():
        # Time input at the top
        Time = st.number_input(
            "Time", 
            value=452.0, 
            help="Elapsed time in seconds since the first transaction in the dataset."
        )

        # Create three columns for V1 to V28
        col1, col2, col3 = st.columns(3)

        with col1:
            V1 = st.number_input(
                "Transaction Behavior Score 1 (V1)", 
                value=-1.359807, 
                help="A feature derived from transaction data patterns."
            )
            V4 = st.number_input(
                "Transaction Behavior Score 4 (V4)", 
                value=1.378155, 
                help="A feature derived from transaction data patterns."
            )
            V7 = st.number_input(
                "Transaction Behavior Score 7 (V7)", 
                value=0.239599, 
                help="A feature derived from transaction data patterns."
            )
            V10 = st.number_input(
                "Transaction Behavior Score 10 (V10)", 
                value=0.090794, 
                help="A feature derived from transaction data patterns."
            )
            V13 = st.number_input(
                "Transaction Behavior Score 13 (V13)", 
                value=-0.991, 
                help="A feature derived from transaction data patterns."
            )
            V16 = st.number_input(
                "Transaction Behavior Score 16 (V16)", 
                value=-0.4704, 
                help="A feature derived from transaction data patterns."
            )
            V19 = st.number_input(
                "Transaction Behavior Score 19 (V19)", 
                value=0.403992, 
                help="A feature derived from transaction data patterns."
            )
            V22 = st.number_input(
                "Transaction Behavior Score 22 (V22)", 
                value=0.277838, 
                help="A feature derived from transaction data patterns."
            )
            V25 = st.number_input(
                "Transaction Behavior Score 25 (V25)", 
                value=0.128539, 
                help="A feature derived from transaction data patterns."
            )

        with col2:
            V2 = st.number_input(
                "Transaction Behavior Score 2 (V2)", 
                value=-0.072781, 
                help="A feature derived from transaction data patterns."
            )
            V5 = st.number_input(
                "Transaction Behavior Score 5 (V5)", 
                value=-0.338321, 
                help="A feature derived from transaction data patterns."
            )
            V8 = st.number_input(
                "Transaction Behavior Score 8 (V8)", 
                value=0.098698, 
                help="A feature derived from transaction data patterns."
            )
            V11 = st.number_input(
                "Transaction Behavior Score 11 (V11)", 
                value=-0.5516, 
                help="A feature derived from transaction data patterns."
            )
            V14 = st.number_input(
                "Transaction Behavior Score 14 (V14)", 
                value=-0.311169, 
                help="A feature derived from transaction data patterns."
            )
            V17 = st.number_input(
                "Transaction Behavior Score 17 (V17)", 
                value=0.20797, 
                help="A feature derived from transaction data patterns."
            )
            V20 = st.number_input(
                "Transaction Behavior Score 20 (V20)", 
                value=0.251412, 
                help="A feature derived from transaction data patterns."
            )
            V23 = st.number_input(
                "Transaction Behavior Score 23 (V23)", 
                value=-0.110473, 
                help="A feature derived from transaction data patterns."
            )
            V26 = st.number_input(
                "Transaction Behavior Score 26 (V26)", 
                value=-0.189114, 
                help="A feature derived from transaction data patterns."
            )

        with col3:
            V3 = st.number_input(
                "Transaction Behavior Score 3 (V3)", 
                value=2.536346, 
                help="A feature derived from transaction data patterns."
            )
            V6 = st.number_input(
                "Transaction Behavior Score 6 (V6)", 
                value=0.462388, 
                help="A feature derived from transaction data patterns."
            )
            V9 = st.number_input(
                "Transaction Behavior Score 9 (V9)", 
                value=0.363787, 
                help="A feature derived from transaction data patterns."
            )
            V12 = st.number_input(
                "Transaction Behavior Score 12 (V12)", 
                value=-0.6178, 
                help="A feature derived from transaction data patterns."
            )
            V15 = st.number_input(
                "Transaction Behavior Score 15 (V15)", 
                value=1.468177, 
                help="A feature derived from transaction data patterns."
            )
            V18 = st.number_input(
                "Transaction Behavior Score 18 (V18)", 
                value=0.0258, 
                help="A feature derived from transaction data patterns."
            )
            V21 = st.number_input(
                "Transaction Behavior Score 21 (V21)", 
                value=-0.018307, 
                help="A feature derived from transaction data patterns."
            )
            V24 = st.number_input(
                "Transaction Behavior Score 24 (V24)", 
                value=0.066928, 
                help="A feature derived from transaction data patterns."
            )
            V27 = st.number_input(
                "Transaction Behavior Score 27 (V27)", 
                value=0.133558, 
                help="A feature derived from transaction data patterns."
            )

        V28 = st.number_input(
            "Transaction Behavior Score 28 (V28)", 
            value=-0.021053, 
            help="A feature derived from transaction data patterns."
        )
        # Amount input at the bottom
        Amount = st.number_input(
            "Transaction Amount", 
            value=149.62, 
            help="The total transaction amount in monetary units."
        )

        # Collect all inputs in a DataFrame
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


# Page 3: Provide Feedback
with tab3:
    st.markdown("""<div class='card'><h3 style="text-align: center;">💬 We Value Your Feedback!</h3></div>""", unsafe_allow_html=True)
        
    name = st.text_input("Name", key="name_input")
    feedback = st.text_area("Message", key="feedback_input")

    if st.button("🚀 Submit Feedback"):
        if name and feedback:
            with open("feedback.txt", "a") as f:
                f.write(f"{datetime.now().date()} - {name}: {feedback}\n")
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please provide both name and feedback.")
    try:
        with open("feedback.txt", "r") as f:
            for line in f.readlines():
                st.markdown(f"<div class='feedback-box'>{line}</div>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.info("No feedback has been provided yet.")
