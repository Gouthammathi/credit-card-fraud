# Credit Card Fraud Detection System

A Streamlit-based web application for detecting fraudulent credit card transactions using machine learning.

## Features

- Interactive web interface
- Real-time fraud detection
- Visual analytics and reporting
- Easy-to-understand results
- Downloadable reports

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd credit-card-fraud
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you're in the project directory
2. Run the Streamlit app:
```bash
streamlit run interface_credit.py
```

3. Access the application at:
   - Local URL: http://localhost:8501
   - Network URL: http://<your-ip>:8501

## Deployment

### Streamlit Cloud Deployment

1. Create a Streamlit Cloud account at https://streamlit.io/cloud
2. Connect your GitHub repository
3. Select the main branch and the interface_credit.py file
4. Add the following environment variables if needed:
   - PYTHON_VERSION: 3.10
5. Deploy the application

### Local Deployment

1. Install all dependencies using requirements.txt
2. Run the application using the command above
3. Use a reverse proxy like Nginx if needed for production deployment

## Data Format

The application expects a CSV file with the following columns:
- Time: Transaction timestamp
- Amount: Transaction amount
- V1-V28: Encrypted transaction features
- Class: Transaction type (0: legitimate, 1: fraud)

## Troubleshooting

If you encounter any issues:

1. Check that all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Ensure you have the correct Python version (3.10 or higher)

3. If you get a "ModuleNotFoundError", try reinstalling the specific package:
```bash
pip install --upgrade <package-name>
```

4. For deployment issues, check the Streamlit Cloud logs for specific error messages

## Support

For any issues or questions, please open an issue in the repository. 