# System Configuration

# Alert Thresholds
ALERT_THRESHOLDS = {
    'fraud_probability': 0.8,
    'high_risk_amount': 1000,
    'alert_cooldown_minutes': 5
}

# Model Settings
MODEL_SETTINGS = {
    'refresh_frequency_minutes': 60,
    'model_path': 'svc_model.pkl'
}

# Notification Settings
EMAIL_SETTINGS = {
    'sender_email': 'your-email@gmail.com',  # Replace with your email
    'recipient_email': 'recipient@example.com',  # Replace with recipient email
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'password': 'your-app-password'  # Replace with your app password
}

# API Settings
API_SETTINGS = {
    'api_key': 'your-api-key',
    'rate_limit': 100
}

# Database Settings
DB_SETTINGS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'fraud_detection',
    'user': 'postgres',
    'password': 'your-password'
}

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FILE = "fraud_detection.log"

# Feature Importance Weights
FEATURE_WEIGHTS = {
    'amount': 0.3,
    'time': 0.2,
    'location': 0.2,
    'merchant': 0.15,
    'device': 0.15
}

# Risk Scoring Parameters
RISK_SCORE_WEIGHTS = {
    'amount': 0.4,
    'time': 0.3,
    'location': 0.2,
    'velocity': 0.1
}

# Performance Metrics Thresholds
PERFORMANCE_THRESHOLDS = {
    'precision': 0.95,
    'recall': 0.90,
    'f1_score': 0.92
} 