# Fraudulent Claim Detection

## Project Overview

This project develops a machine learning solution for Global Insure, a leading insurance company, to automatically identify fraudulent insurance claims and minimize financial losses. The system uses historical claim data and customer profiles to predict the likelihood of fraud before claims are approved, replacing time-consuming manual inspections with an efficient data-driven approach.

## Problem Statement

Global Insure processes thousands of claims annually, with a significant percentage turning out to be fraudulent, resulting in considerable financial losses. The current manual inspection process for identifying fraudulent claims is time-consuming and inefficient, often detecting fraud too late after significant payouts have already been made. This project aims to improve the fraud detection process using data-driven insights to classify claims early in the approval process.

## Business Objective

Build a predictive model to classify insurance claims as either fraudulent or legitimate based on historical claim details and customer profiles. The model leverages features such as claim amounts, customer profiles, and claim types to predict fraud likelihood and optimize the overall claims handling process.

## Dataset

The insurance claims dataset contains 40 columns and 1,000 rows with the following key features:

### Customer Information
- **months_as_customer**: Duration in months that a customer has been with the insurance company
- **age**: Age of the insured person
- **policy_number**: Unique identifier for each insurance policy
- **policy_bind_date**: Date when the insurance policy was initiated
- **policy_state**: State where the insurance policy is applicable
- **policy_annual_premium**: Yearly cost of the insurance policy

### Claim Details
- **incident_date**: Date when the incident or accident occurred
- **incident_type**: Type of incident (collision, theft, etc.)
- **collision_type**: Specific type of collision
- **incident_severity**: Severity level of the incident
- **total_claim_amount**: Total amount claimed for the incident
- **injury_claim**: Amount claimed for injuries
- **property_claim**: Amount claimed for property damage
- **vehicle_claim**: Amount claimed for vehicle damage

### Vehicle Information
- **auto_make**: Manufacturer of the insured vehicle
- **auto_model**: Specific model of the insured vehicle
- **auto_year**: Year of manufacture of the insured vehicle

### Financial Information
- **capital_gains**: Profit earned from asset sales
- **capital_loss**: Loss incurred from asset sales

### Investigation Details
- **bodily_injuries**: Number of bodily injuries resulting from the incident
- **witnesses**: Number of witnesses present at the scene
- **police_report_available**: Whether a police report is available

### Target Variable
- **fraud_reported**: Whether the claim was reported as fraudulent (target variable)

## Methodology

### 1. Data Preparation and Cleaning
- Data loading and initial exploration
- Handling missing values and data quality issues
- Data type conversions and formatting

### 2. Data Splitting
- Training and validation split (70-30 ratio)
- Stratified sampling to maintain class balance

### 3. Exploratory Data Analysis (EDA)
- Statistical analysis of features
- Distribution analysis of fraudulent vs. legitimate claims
- Correlation analysis and feature relationships
- Visualization of key patterns and insights

### 4. Feature Engineering
- Date feature extraction (year, month, day of week)
- Creation of derived features (days from policy date)
- Categorical variable encoding
- Feature scaling and normalization

### 5. Data Balancing
- RandomOverSampler implementation to handle class imbalance
- Resampling of training data for improved model performance

### 6. Model Building
Two primary models were implemented and evaluated:

#### Logistic Regression
- Recursive Feature Elimination with Cross-Validation (RFECV)
- Variance Inflation Factor (VIF) analysis for multicollinearity
- Statistical significance testing using statsmodels

#### Random Forest Classifier
- Feature importance analysis
- Hyperparameter tuning using GridSearchCV
- Class weight balancing for imbalanced data

### 7. Model Evaluation
- Accuracy metrics
- Confusion matrix analysis
- ROC curve and AUC score
- Precision-Recall curve
- Classification report with precision, recall, and F1-score

## Technologies Used

### Programming Language
- Python 3.x

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and tools
- **statsmodels**: Statistical modeling and analysis
- **imblearn**: Handling imbalanced datasets

### Visualization
- **matplotlib**: Basic plotting and visualization
- **seaborn**: Statistical data visualization

### Machine Learning Algorithms
- **Logistic Regression**: Linear classification with feature selection
- **Random Forest**: Ensemble method with feature importance analysis

### Feature Selection and Preprocessing
- **StandardScaler**: Feature scaling and normalization
- **RFECV**: Recursive feature elimination with cross-validation
- **RandomOverSampler**: Handling class imbalance

## Key Findings

The analysis revealed several important insights:

1. **Feature Importance**: Certain claim characteristics and customer profiles are strong predictors of fraudulent behavior
2. **Temporal Patterns**: Time-based features (days from policy date, incident timing) provide valuable predictive power
3. **Model Performance**: Both logistic regression and random forest models demonstrated effective fraud detection capabilities
4. **Class Imbalance**: Addressing class imbalance through oversampling significantly improved model performance

## Installation and Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels imbalanced-learn
```

### Running the Analysis
1. Clone the repository
2. Open the Jupyter notebook: `Fraudulent_Claim_Detection_Bhuvanshree_B_Jeffrin_Chittilappilly.ipynb`
3. Run all cells sequentially to reproduce the analysis
4. Review the model performance metrics and visualizations

## File Structure
```
Fraudulent_Claim_Detection/
├── README.md
├── .gitignore
├── Fraudulent_Claim_Detection_Bhuvanshree_B_Jeffrin_Chittilappilly/
│   ├── Fraudulent_Claim_Detection_Bhuvanshree_B_Jeffrin_Chittilappilly.ipynb
│   ├── Fraudulent_Claim_Detection_Bhuvanshree_B_Jeffrin_Chittilappilly.pdf
│   └── Fraudulent_Claim_Detection_Bhuvanshree_B_Jeffrin_Chittilappilly.key
└── Fraudulent_Claim_Detection_Bhuvanshree_B_Jeffrin_Chittilappilly.zip
```

## Results and Impact

The developed fraud detection system provides:
- Early identification of potentially fraudulent claims
- Reduced manual inspection workload
- Minimized financial losses from fraudulent payouts
- Optimized claims processing workflow
- Data-driven insights for continuous improvement

## Future Enhancements

Potential improvements include:
- Implementation of additional machine learning algorithms (XGBoost, Neural Networks)
- Real-time fraud detection pipeline
- Integration with existing insurance management systems
- Advanced feature engineering techniques
- Continuous model monitoring and retraining

## Contributors

- Bhuvanshree B Jeffrin Chittilappilly

## Contact

For questions or collaboration opportunities, please reach out through the repository's issue tracker.
