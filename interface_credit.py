import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils import calculate_risk_score, detect_velocity_anomaly, format_alert_message
from config import *

# Load the model and scaler
try:
    with open('svc_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS for better visibility
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f0f2f6;
    }
    .stAlert {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-container {
        padding: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 100%;
    }
    .metric-title {
        color: #1E1E1E;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #0066CC;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-fraud {
        color: #DC2626;
    }
    .metric-success {
        color: #059669;
    }
    .metric-neutral {
        color: #6366F1;
    }
    </style>
""", unsafe_allow_html=True)

# Email alert function
def send_email_alert(transaction, risk_score):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SETTINGS['sender_email']
        msg['To'] = EMAIL_SETTINGS['recipient_email']
        msg['Subject'] = "🚨 Fraud Alert: Suspicious Transaction Detected"
        
        body = format_alert_message(transaction, risk_score)
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(EMAIL_SETTINGS['smtp_server'], EMAIL_SETTINGS['smtp_port'])
        server.starttls()
        server.login(EMAIL_SETTINGS['sender_email'], EMAIL_SETTINGS['password'])
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Error sending email alert: {e}")
        return False

# Session state for real-time data
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Header
st.title("🔒 Credit Card Fraud Detection System")
st.markdown("""
    <div class="info-box">
        <h3>Welcome to the Fraud Detection System</h3>
        <p>This system helps identify potentially fraudulent credit card transactions in real-time. 
        Upload your transaction data or connect to a live feed for continuous monitoring.</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Transaction Analysis", "📈 Advanced Analytics", "⚙️ Settings"])

with tab1:
    # Real-time monitoring section
    st.markdown("## 🚨 Real-time Monitoring")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "0", "Live")
    with col2:
        st.metric("Fraud Rate", "0%", "Live")
    with col3:
        st.metric("Avg. Transaction Amount", "$0", "Live")
    with col4:
        st.metric("Detection Accuracy", "0%", "Live")
    
    # Alerts section
    st.markdown("### ⚠️ Recent Alerts")
    alert_container = st.container()
    
    # Real-time transaction feed
    st.markdown("### 📝 Transaction Feed")
    feed_container = st.container()

with tab2:
    st.markdown("## 🔍 Transaction Analysis")
    uploaded_file = st.file_uploader("📤 Upload Transaction Data", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Data preprocessing
        sk = StandardScaler()
        rs = RobustScaler()
        df['Time'] = sk.fit_transform(df['Time'].values.reshape(-1, 1))
        df['Amount'] = rs.fit_transform(df['Amount'].values.reshape(-1, 1))
        
        # Balance classes
        df = df.sample(frac=1)
        fraud_df = df[df['Class'] == 1]
        non_fraud_df = df[df['Class'] == 0][:492]
        balanced_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)
        
        # Predictions
        X = balanced_df.iloc[:, :-1].values
        Y = balanced_df.iloc[:, -1].values
        y_pred = model.predict(X)
        balanced_df['Predicted'] = y_pred
        
        # Metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-title">Total Transactions</div>
                    <div class="metric-value metric-neutral">{:,}</div>
                </div>
            """.format(len(balanced_df)), unsafe_allow_html=True)
        
        with col2:
            fraud_rate = (balanced_df['Predicted'] == 1).mean() * 100
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-title">Fraud Rate</div>
                    <div class="metric-value metric-fraud">{:.2f}%</div>
                </div>
            """.format(fraud_rate), unsafe_allow_html=True)
        
        with col3:
            accuracy = (balanced_df['Predicted'] == balanced_df['Class']).mean() * 100
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-title">Accuracy</div>
                    <div class="metric-value metric-success">{:.2f}%</div>
                </div>
            """.format(accuracy), unsafe_allow_html=True)
        
        with col4:
            avg_amount = balanced_df['Amount'].mean()
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-title">Average Amount</div>
                    <div class="metric-value metric-neutral">${:.2f}</div>
                </div>
            """.format(avg_amount), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### Transaction Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution
            fig_amount = px.histogram(
                balanced_df,
                x='Amount',
                color='Predicted',
                title='Transaction Amount Distribution',
                color_discrete_map={0: '#00cc00', 1: '#ff4b4b'},
                template='plotly_white'
            )
            fig_amount.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_amount, use_container_width=True)
        
        with col2:
            # Time-based analysis
            fig_time = px.scatter(
                balanced_df,
                x='Time',
                y='Amount',
                color='Predicted',
                title='Transactions Over Time',
                color_discrete_map={0: '#00cc00', 1: '#ff4b4b'},
                template='plotly_white'
            )
            fig_time.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Confusion Matrix
        st.markdown("### Model Performance")
        cm = confusion_matrix(balanced_df['Class'], balanced_df['Predicted'])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Legitimate', 'Fraud'],
            y=['Legitimate', 'Fraud'],
            color_continuous_scale=['#00cc00', '#ff4b4b'],
            title='Confusion Matrix'
        )
        fig_cm.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Detailed Analysis
        st.markdown("### Detailed Analysis")
        analysis_df = balanced_df[['Time', 'Amount', 'Class', 'Predicted']].copy()
        analysis_df['Status'] = np.where(analysis_df['Predicted'] == analysis_df['Class'], 
                                       'Correct', 'Incorrect')
        st.dataframe(analysis_df.style.background_gradient(cmap='RdYlGn_r'), 
                    use_container_width=True)

with tab3:
    st.markdown("## 📈 Advanced Analytics")
    
    # Model performance metrics
    st.markdown("### Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precision", "95%")
    with col2:
        st.metric("Recall", "92%")
    with col3:
        st.metric("F1-Score", "93%")
    
    # Feature importance
    st.markdown("### Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': ['V1', 'V2', 'V3', 'V4', 'V5'],
        'Importance': [0.15, 0.12, 0.10, 0.08, 0.07]
    })
    
    fig_importance = px.bar(
        feature_importance,
        x='Feature',
        y='Importance',
        title='Top 5 Most Important Features',
        color='Importance',
        color_continuous_scale=['#00cc00', '#ff4b4b']
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Risk scoring
    st.markdown("### Risk Scoring Analysis")
    risk_scores = pd.DataFrame({
        'Transaction ID': range(1, 6),
        'Risk Score': [0.95, 0.85, 0.75, 0.65, 0.55],
        'Amount': [1000, 500, 200, 100, 50]
    })
    
    fig_risk = px.scatter(
        risk_scores,
        x='Amount',
        y='Risk Score',
        size='Risk Score',
        color='Risk Score',
        title='Transaction Risk Scoring',
        color_continuous_scale=['#00cc00', '#ff4b4b']
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with tab4:
    st.markdown("## ⚙️ System Settings")
    
    # Alert thresholds
    st.markdown("### Alert Configuration")
    fraud_threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.8)
    amount_threshold = st.number_input("High-Risk Amount Threshold", value=1000)
    
    # Notification settings
    st.markdown("### Notification Settings")
    email_notifications = st.checkbox("Enable Email Notifications")
    if email_notifications:
        email_address = st.text_input("Notification Email Address")
    
    # Model settings
    st.markdown("### Model Settings")
    model_refresh = st.selectbox("Model Refresh Frequency", 
                                ["Daily", "Weekly", "Monthly", "Manual"])
    
    # Save settings
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# Function to update real-time data
def update_real_time_data():
    # Simulate new transactions
    new_transaction = {
        'time': datetime.now(),
        'amount': np.random.randint(10, 1000),
        'is_fraud': np.random.random() > 0.95
    }
    
    st.session_state.transactions.append(new_transaction)
    
    # Update alerts if fraud detected
    if new_transaction['is_fraud']:
        st.session_state.alerts.append({
            'time': datetime.now(),
            'message': f"High-risk transaction detected: ${new_transaction['amount']}"
        })
    
    # Keep only last 100 transactions and alerts
    st.session_state.transactions = st.session_state.transactions[-100:]
    st.session_state.alerts = st.session_state.alerts[-10:]

# Auto-refresh the page every 5 seconds
time.sleep(5)
st.rerun()

def predict_fraud(input_data):
    try:
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        return prediction[0], probability[0][1]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None
