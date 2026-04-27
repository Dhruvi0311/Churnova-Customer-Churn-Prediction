# Customer Churn Prediction and Analytics Dashboard

Customer churn refers to the loss of customers over time and is one of the most critical business challenges in subscription-based industries such as telecommunications, SaaS, and financial services. Retaining existing customers is significantly more cost-effective than acquiring new ones, making churn prediction an essential analytical task.

This project focuses on building a machine learning-driven system that predicts customer churn and provides analytical insights through a structured and interactive dashboard.

---

## Project Overview

The system is designed to:

- Predict the probability of customer churn using machine learning
- Estimate Customer Lifetime Value (CLV)
- Identify key factors influencing churn
- Provide visual insights for decision-making
- Deliver actionable recommendations for customer retention

---

## Application Interface

<p align="center">
<img src="https://github.com/Dhruvi0311/Churnova-Customer-Churn-Prediction/blob/main/Images/dashboard.png" width="850">
</p>

---

## Prediction Output

<p align="center">
<img src="https://github.com/Dhruvi0311/Churnova-Customer-Churn-Prediction/blob/main/Images/result.png" width="850">
</p>

---

## System Visualizations

<p align="center">
<img src="https://github.com/Dhruvi0311/Churnova-Customer-Churn-Prediction/blob/main/Images/analysis.png" width="850">
</p>

---

## Data Analysis

Exploratory data analysis was performed to understand customer behavior patterns and identify relationships between features and churn.

### Churn vs Tenure

<p align="center">
<img src="https://github.com/Dhruvi0311/Churnova-Customer-Churn-Prediction/blob/main/Images/tenure-churn.png" width="700">
</p>

Customers with longer tenure exhibit significantly lower churn rates, indicating increased loyalty over time.

---

### Monthly Charges Distribution

<p align="center">
<img src="https://github.com/Dhruvi0311/Churnova-Customer-Churn-Prediction/blob/main/Images/monthlycharges.png" width="500">
</p>

Customers with higher monthly charges are more likely to churn, suggesting pricing sensitivity.

---

### Internet Service and Contract Type

<p align="center">
<img src="https://github.com/Dhruvi0311/Churnova-Customer-Churn-Prediction/blob/main/Images/internetservice-contract.png" width="600">
</p>

Customers with month-to-month contracts and fiber optic services show higher churn probability.

---

## Machine Learning Model

A Random Forest Classifier was used for churn prediction.

### Model Characteristics

- Handles non-linear relationships effectively  
- Robust to noise and feature interactions  
- Suitable for structured tabular data  

### Performance Metrics

- ROC-AUC Score: ~0.85  
- F1 Score: ~0.62  

---

## Feature Importance

<p align="center">
<img src="https://github.com/Dhruvi0311/Churnova-Customer-Churn-Prediction/blob/main/Images/model_feat_imp.png" width="700">
</p>

Feature importance analysis highlights the most influential variables driving churn predictions.

---

## Model Insights

The model provides the following analytical outputs:

- Identification of high-risk customers  
- Understanding of feature-level impact  
- Probability-based churn classification  
- Customer segmentation based on risk and value  

---

## Customer Value Analysis

Customer Lifetime Value (CLV) is estimated using:

- Monthly charges  
- Expected customer lifespan  

This helps in identifying high-value customers at risk, enabling targeted retention strategies.

---

## Business Recommendations

Based on model outputs and analysis:

- Encourage long-term contracts to reduce churn  
- Provide incentives for high-risk customers  
- Improve engagement for customers with fewer services  
- Promote automated payment methods  
- Monitor high-value customers with elevated churn probability  

---

## Technology Stack

- Python  
- Flask  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib  
- HTML and CSS  

---

## Project Structure
.
├── Images/                             : contains images
├── static/                             : plots to show gauge chart, hazard and survival curve, shap values in Flask App 
│   └── images/
│       ├── hazard.png
│       ├── surv.png
│       ├── shap.png
│       └── new_plot.png
├── templates/                          : contains html template for flask app
│   └── index.html
├── Customer Survival Analysis.ipynb    : Survival Analysis kaplan-Meier curve, log-rank test and Cox-proportional Hazard model
├── Exploratory Data Analysis.ipynb     : Data Analysis to understand customer data
├── Churn Prediction Model.ipynb        : Random Forest model to predict customer churn
├── app.py                              : Flask App
├── app-pic.png                         : Final App image  
├── explainer.bz2                       : Shap Explainer
├── model.pkl                           : Random Forest model
├── survivemodel.pkl                    : Cox-proportional Hazard model
├── requirements.txt                    : requirements to run this model
├── Procfile                            : procfile for app deployment
├── LICENSE.md                          : MIT License
└── README.md                           : Report

## Installation and Execution

Clone the repository:


git clone https://github.com/Dhruvi0311/Churnova-Customer-Churn-Prediction.git

cd Churnova-Customer-Churn-Prediction


Install dependencies:


pip install -r requirements.txt


Run the application:


python app.py


Access the application at:


http://127.0.0.1:5000


---

## Conclusion

This project demonstrates how machine learning models can be integrated with interactive interfaces to create decision-support systems. It not only predicts customer churn but also provides insights that help businesses take informed actions to improve customer retention and maximize long-term value.