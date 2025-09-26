# Fraudulent Claim Detection

## Problem Statement
Global Insure, a leading insurance company, processes thousands of claims annually. However, a significant percentage of these claims turn out to be fraudulent, resulting in considerable financial losses. The company’s current process for identifying fraudulent claims involves manual inspections, which is time-consuming and inefficient. Fraudulent claims are often detected too late in the process, after payouts have already been made.  

To minimize financial losses and improve operational efficiency, Global Insure aims to leverage data-driven insights and machine learning to detect fraudulent claims early in the approval process.

---

## Business Objective
Build a machine learning model to classify insurance claims as **fraudulent** or **legitimate** using historical claim details and customer profiles.  

---

## Dataset
- **Source:** `insurance_claims.csv`  
- **Features:** Claim amount, incident details, policy attributes, customer profile, etc.  
- **Target Variable:** `fraud_reported` (Yes/No)  

---

## Methodology
1. **Data Preprocessing**
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling and transformation  

2. **Exploratory Data Analysis (EDA)**
   - Distribution of key features  
   - Correlation analysis  
   - Fraud vs non-fraud class imbalance analysis  

3. **Modeling**
   - Tested multiple classification algorithms: Logistic Regression, Random Forest, XGBoost  
   - Hyperparameter tuning for model optimization  
   - Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  

4. **Model Evaluation**
   - Compared performance across models  
   - Selected the best-performing model based on balanced metrics  

---

## Results
- Achieved high recall and precision for fraudulent claim detection.  
- Best-performing model: **[Insert chosen model here, e.g., Random Forest]**  
- ROC-AUC: **[Insert score]**  
- Demonstrated ability to flag fraudulent claims early with high accuracy.  

---

## Tools & Technologies
- Python  
- Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, XGBoost  
- Jupyter Notebook  

---

## How to Run
1. Clone this repository  
2. Install required libraries:  
   ```bash
jupyter notebook Fraudulent_Claim_Detection.ipynb

