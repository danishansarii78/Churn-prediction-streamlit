# Churn-prediction-streamlit
🎯 Project Overview
This project analyzes customer churn data to predict which customers are likely to leave a telecommunications service. Using machine learning, we can identify at-risk customers and help businesses take proactive retention measures.

📊 Key Insights
Dataset: 7,032 customers with 31 features

Accuracy: 79% with Random Forest Classifier

Business Impact: Identify 48% of potential churners early

🚀 Quick Start
Prerequisites
bash
pip install pandas numpy scikit-learn jupyter
Run the Project
bash
# Clone this repository
git clone https://github.com/danishansarii78/Churn-prediction-streamlit.git

# Navigate to project directory
cd Churn-prediction-streamlit

# Launch Jupyter Notebook
jupyter notebook
Open CUSTOMER_CHURN.ipynb and run all cells to see the magic! ✨

📈 Model Performance
Confusion Matrix
text
[[926 107]  ← Correctly predicted non-churners: 926
 [193 181]] ← Correctly predicted churners: 181
Classification Report
Class	Precision	Recall	F1-Score	Support
No Churn (0)	83%	90%	86%	1,033
Churn (1)	63%	48%	55%	374
Overall Accuracy	79%			1,407
🛠️ Technical Implementation
Data Preprocessing
python
# Key steps in our pipeline:
1. ✅ Handle missing values in TotalCharges
2. ✅ Convert Churn to binary (Yes→1, No→0)
3. ✅ One-Hot Encoding for categorical variables
4. ✅ Train-Test Split (80-20)
Model Architecture
Algorithm: Random Forest Classifier

Features: 30 engineered features after encoding

Target: Churn (Binary Classification)

Saved Artifacts
churn_model.pkl - Trained Random Forest model

churn_features.pkl - Feature columns for prediction

💡 Business Applications
🎯 Use Cases
Customer Retention: Identify high-risk customers for targeted offers

Resource Allocation: Focus retention efforts where they matter most

Product Insights: Understand which features drive customer satisfaction

📊 Actionable Insights
The model can identify nearly half of potential churners

79% overall accuracy in predicting customer behavior

Can be integrated into CRM systems for real-time predictions

🗂️ Project Structure
text
customer-churn-prediction/
│
├── CUSTOMER_CHURN.ipynb          # Main analysis notebook
├── churn_model.pkl              # Trained model (generated)
├── churn_features.pkl           # Feature columns (generated)
├── requirements.txt             # Dependencies
└── README.md                   # This file
🎮 Interactive Demo
Want to test the model with your own data? Here's how:

python
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('churn_model.pkl', 'rb'))
features = pickle.load(open('churn_features.pkl', 'rb'))

# Create sample input (replace with your data)
sample_data = {
    'tenure': 12,
    'MonthlyCharges': 70.5,
    'TotalCharges': 850.0,
    'Contract_Two year': 1,
    # ... add all other features
}

# Make prediction
prediction = model.predict([sample_data])
probability = model.predict_proba([sample_data])

print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Confidence: {probability[0][1]:.2%}")
📊 Results Interpretation
🎯 What the Metrics Mean
Precision (63%): When we predict churn, we're correct 63% of the time

Recall (48%): We catch 48% of all actual churn cases

F1-Score (55%): Balanced measure of precision and recall

🔍 Model Strengths
✅ Excellent at identifying loyal customers (90% recall for class 0)

✅ Good overall accuracy for business decisions

✅ Handles imbalanced data reasonably well

🚧 Areas for Improvement
⚠️ Could improve churn detection rate (currently 48%)

⚠️ Feature engineering could be enhanced

⚠️ Consider handling class imbalance with techniques like SMOTE
