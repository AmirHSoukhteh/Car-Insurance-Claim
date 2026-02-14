# Car Insurance Claim Prediction

Predicting insurance claims using machine learning - Decision Trees and Support Vector Machines.

**Course:** Data Mining  
**Academy:** Hamrah Aval Academy

---

## About

This project uses machine learning to predict whether a car insurance customer will file a claim based on their demographic information, driving history, and vehicle details. We compare Decision Tree and SVM classifiers, optimize their performance, and identify the most important features for prediction.

## Project Structure

```
Car-Insurance-Claim/
├── data/
│   ├── raw/
│   │   └── Car_Insurance_Claim 1.csv
│   └── processed/
├── notebooks/
│   └── car_insurance_analysis.ipynb
├── src/
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The dataset contains **10,000 records** with the following features:

### Demographic Features:
- **AGE**: Age group (16-25, 26-39, 40-64, 65+)
- **GENDER**: male/female
- **RACE**: majority/minority
- **MARRIED**: Marital status (0/1)
- **CHILDREN**: Number of children
- **EDUCATION**: Education level (none, high school, university)
- **INCOME**: Income class (poverty, working class, upper class)

### Vehicle & Driving Features:
- **DRIVING_EXPERIENCE**: Years of driving (0-9y, 10-19y, 20-29y, 30y+)
- **VEHICLE_OWNERSHIP**: Owns vehicle (0/1)
- **VEHICLE_YEAR**: before 2015 / after 2015
- **VEHICLE_TYPE**: sedan / sports car
- **ANNUAL_MILEAGE**: Miles driven per year
- **POSTAL_CODE**: Location code

### Risk Features:
- **CREDIT_SCORE**: Credit score (0-1 normalized)
- **SPEEDING_VIOLATIONS**: Number of violations
- **DUIS**: Driving under influence incidents
- **PAST_ACCIDENTS**: Previous accidents

### Target Variable:
- **OUTCOME**: Filed claim (0=No, 1=Yes)

## What's Inside

### 1. Data Preparation
- Load dataset and remove unnecessary ID column
- Check for missing values and data types
- Identify numeric vs categorical variables

### 2. Exploratory Data Analysis
- Statistical summaries
- Distribution analysis
- Outcome variable balance check

### 3. Data Cleaning
- **Outlier Removal**: Using IQR (Interquartile Range) method on CREDIT_SCORE
- **Comparison Analysis**: Compare claim vs no-claim groups

### 4. Data Balancing
- Balance dataset using undersampling
- Ensure equal representation of both outcome classes

### 5. Feature Engineering
- **One-Hot Encoding**: Convert categorical variables to numeric
- Create binary features for ML algorithms

### 6. Model Building
- **Decision Tree Classifier**: Tree-based model with interpretable rules
- **Support Vector Machine (SVM)**: Kernel-based classifier for complex patterns

### 7. Model Evaluation
- Accuracy scores
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- Performance comparison

### 8. Feature Importance
- Identify most predictive features
- Rank variables by importance
- Visualize top contributors

### 9. Hyperparameter Optimization
- GridSearchCV for SVM optimization
- Test combinations of:
  - Kernel: linear, rbf
  - C: 0.01, 0.1, 1, 10, 100
  - Gamma: 0.1, 0.01, 0.001, auto

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/car-insurance-claim.git
cd car-insurance-claim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
```bash
jupyter notebook notebooks/car_insurance_analysis.ipynb
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

See `requirements.txt` for specific versions.

## Results

### Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Decision Tree | ~78% | Fast, interpretable |
| SVM (Default) | ~76% | Good baseline |
| SVM (Optimized) | ~80% | Best performance after GridSearch |

### Top Predictive Features

1. **VEHICLE_OWNERSHIP** (28.8%) - Strongest predictor
2. **PAST_ACCIDENTS** (19.3%) - Second most important
3. **VEHICLE_YEAR** (15.6%) - Older vehicles correlate with claims
4. **POSTAL_CODE** (9.7%) - Location matters
5. **DRIVING_EXPERIENCE** (9.5%) - Experience level impacts claims

## Key Findings

✅ **Vehicle ownership** is the strongest predictor of insurance claims  
✅ **Past accident history** significantly correlates with future claims  
✅ **Older vehicles** (before 2015) show higher claim rates  
✅ **Location** (postal code) influences claim probability  
✅ **GridSearch optimization** improved SVM performance by ~4%

## Machine Learning Techniques Used

- **Classification Algorithms**: Decision Trees, Support Vector Machines
- **Data Preprocessing**: Outlier removal (IQR), one-hot encoding, data balancing
- **Model Evaluation**: Cross-validation, confusion matrices, classification reports
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Feature Engineering**: One-hot encoding for categorical variables
- **Feature Selection**: Feature importance ranking from Decision Tree

## Business Applications

This model can help insurance companies:
1. **Risk Assessment**: Identify high-risk customers
2. **Premium Pricing**: Adjust premiums based on risk factors
3. **Fraud Detection**: Flag suspicious claim patterns
4. **Customer Segmentation**: Group customers by risk profile
5. **Resource Allocation**: Focus on high-risk policyholders

## Future Improvements

- Try ensemble methods (Random Forest, XGBoost, Gradient Boosting)
- Implement deep learning (Neural Networks)
- Add more features (weather data, traffic patterns)
- Use SMOTE for more sophisticated balancing
- Deploy model as web API
- A/B testing in production environment

## License

MIT License - free to use for educational purposes.

## Contact

For questions or feedback, please open an issue or contact via email.

---

**Note:** This is an educational project for the Data Mining course at Hamrah Aval Academy.
