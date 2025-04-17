import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import *

# Set up logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOG_FILE
)
logger = logging.getLogger(__name__)

def calculate_risk_score(transaction):
    """
    Calculate risk score for a transaction based on multiple factors
    """
    try:
        # Base risk score from model prediction
        base_score = transaction.get('fraud_probability', 0)
        
        # Amount-based risk
        amount_risk = min(transaction['amount'] / HIGH_RISK_AMOUNT_THRESHOLD, 1)
        
        # Time-based risk (transactions at unusual hours)
        hour = transaction['time'].hour
        time_risk = 0.5 if 0 <= hour <= 5 else 0.1
        
        # Calculate weighted risk score
        risk_score = (
            base_score * RISK_SCORE_WEIGHTS['amount'] +
            amount_risk * RISK_SCORE_WEIGHTS['amount'] +
            time_risk * RISK_SCORE_WEIGHTS['time']
        )
        
        return min(risk_score, 1.0)  # Cap at 1.0
        
    except Exception as e:
        logger.error(f"Error calculating risk score: {e}")
        return 0.5  # Default risk score

def detect_velocity_anomaly(transactions, window_minutes=60):
    """
    Detect if there are too many transactions in a short time window
    """
    try:
        if len(transactions) < 2:
            return False
            
        # Get transactions within the time window
        recent_transactions = [
            t for t in transactions
            if t['time'] > datetime.now() - timedelta(minutes=window_minutes)
        ]
        
        # Calculate velocity metrics
        transaction_count = len(recent_transactions)
        total_amount = sum(t['amount'] for t in recent_transactions)
        
        # Define thresholds
        count_threshold = 10  # Maximum transactions per hour
        amount_threshold = 5000  # Maximum amount per hour
        
        return (transaction_count > count_threshold or 
                total_amount > amount_threshold)
                
    except Exception as e:
        logger.error(f"Error in velocity anomaly detection: {e}")
        return False

def format_alert_message(transaction, risk_score):
    """
    Format alert message for notifications
    """
    try:
        return (
            f"🚨 High-Risk Transaction Alert\n"
            f"Time: {transaction['time']}\n"
            f"Amount: ${transaction['amount']:.2f}\n"
            f"Risk Score: {risk_score:.2%}\n"
            f"Transaction ID: {transaction.get('id', 'N/A')}"
        )
    except Exception as e:
        logger.error(f"Error formatting alert message: {e}")
        return "Alert: Suspicious transaction detected"

def calculate_performance_metrics(y_true, y_pred):
    """
    Calculate model performance metrics
    """
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return None

def preprocess_transaction(transaction):
    """
    Preprocess transaction data for model input
    """
    try:
        # Normalize amount
        transaction['amount'] = np.log1p(transaction['amount'])
        
        # Extract time features
        transaction['hour'] = transaction['time'].hour
        transaction['day_of_week'] = transaction['time'].weekday()
        
        return transaction
    except Exception as e:
        logger.error(f"Error preprocessing transaction: {e}")
        return None

def validate_transaction_data(transaction):
    """
    Validate transaction data before processing
    """
    try:
        required_fields = ['time', 'amount', 'id']
        return all(field in transaction for field in required_fields)
    except Exception as e:
        logger.error(f"Error validating transaction data: {e}")
        return False 