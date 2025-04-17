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

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .fraud-metric {
        font-size: 24px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .legitimate-metric {
        font-size: 24px;
        font-weight: bold;
        color: #00cc00;
    }
    .stProgress > div > div > div > div {
        background-color: #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🔒 Credit Card Fraud Detection System")
st.markdown("""
    This application uses machine learning to detect fraudulent credit card transactions.
    Upload your transaction data to get real-time fraud detection results with visual analytics.
""")

# Create two columns for the layout
col1, col2 = st.columns([2, 1])

with col2:
    # Upload CSV
    uploaded_file = st.file_uploader("📤 Upload Transaction Data (CSV)", type=['csv'])

# Load pickled model
@st.cache_resource
def load_model():
    with open('svc_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    with col1:
        st.subheader("📊 Sample of Original Data")
        st.dataframe(df.head(), use_container_width=True)

    try:
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

        # Create metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            total_transactions = len(balanced_df)
            st.metric("Total Transactions", f"{total_transactions:,}")
            
        with metrics_col2:
            fraud_count = len(balanced_df[balanced_df['Predicted'] == 1])
            fraud_percentage = (fraud_count / total_transactions) * 100
            st.metric("Detected Fraud", f"{fraud_count:,} ({fraud_percentage:.1f}%)", 
                     delta_color="inverse")
            
        with metrics_col3:
            legitimate_count = len(balanced_df[balanced_df['Predicted'] == 0])
            legitimate_percentage = (legitimate_count / total_transactions) * 100
            st.metric("Legitimate Transactions", f"{legitimate_count:,} ({legitimate_percentage:.1f}%)")

        # Visualization section
        st.subheader("📈 Visual Analytics")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Pie chart for fraud distribution
            fig_pie = px.pie(
                values=[fraud_count, legitimate_count],
                names=['Fraud', 'Legitimate'],
                title='Transaction Distribution',
                color_discrete_sequence=['#ff4b4b', '#00cc00']
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with viz_col2:
            # Confusion matrix
            cm = confusion_matrix(Y, y_pred)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual"),
                x=['Legitimate', 'Fraud'],
                y=['Legitimate', 'Fraud'],
                color_continuous_scale=['#00cc00', '#ff4b4b'],
                title='Confusion Matrix'
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Transaction amount distribution
        st.subheader("💰 Transaction Amount Distribution")
        amount_col1, amount_col2 = st.columns(2)

        with amount_col1:
            # Original amounts before scaling
            original_amounts = df['Amount'].values
            fig_dist = px.histogram(
                x=original_amounts,
                color=df['Class'],
                nbins=50,
                title='Transaction Amount Distribution',
                labels={'x': 'Amount', 'color': 'Class'},
                color_discrete_map={0: '#00cc00', 1: '#ff4b4b'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Results table with color coding
        st.subheader("🔍 Detailed Results")
        
        # Style the dataframe
        def color_fraud(val):
            color = '#ff4b4b' if val == 1 else '#00cc00'
            return f'background-color: {color}; color: white'

        styled_df = balanced_df.style.applymap(
            color_fraud, 
            subset=['Predicted']
        )
        
        st.dataframe(styled_df, use_container_width=True)

        # Download section
        st.subheader("📥 Download Results")
        csv = balanced_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Complete Analysis (CSV)",
            data=csv,
            file_name='fraud_detection_results.csv',
            mime='text/csv',
            help="Download the complete analysis including predictions"
        )

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
else:
    # Show sample visualization when no file is uploaded
    st.info("👆 Please upload a CSV file to begin the fraud detection analysis.")
    st.markdown("""
        ### Expected CSV Format:
        - Time: Transaction timestamp
        - Amount: Transaction amount
        - V1-V28: Transformed features
        - Class: Transaction class (0: legitimate, 1: fraud)
    """)
