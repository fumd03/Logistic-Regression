                      Hearing Test Analysis Using Age and Physical Score

Overview

This project analyzes the relationship between **age**, **physical condition score**, and **hearing test results** using data analysis and machine learning techniques.
The goal is to identify patterns that influence hearing test outcomes and understand how **age and physical health indicators** are related to hearing performance.

Objectives

* Explore how age affects hearing test results
* Analyze the impact of physical score on hearing ability
* Identify patterns and trends in the dataset
* Build a machine learning model to predict hearing test results
* Evaluate model performance using standard metrics

Dataset

The dataset contains three main variables:

Features:
**age** → Age of the individual
**physical_score** → A numerical score representing physical health condition

Target:

**test_result** → Hearing test outcome ( Normal / Impaired or 0 / 1)

Methodology

This project follows a standard machine learning pipeline:

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering (if needed)
5. Model Training
6. Model Evaluation
7. Insight Extraction


Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Jupyter Notebook

Exploratory Data Analysis (EDA)

The following analyses were performed:

* Distribution of age
* Relationship between physical score and test result
* Age vs hearing test outcome patterns
* Correlation between features


Machine Learning Models

The following models were used:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix


How to Run This Project

git clone https://github.com/your-username/hearing-analysis.git
cd hearing-analysis
pip install -r requirements.txt
jupyter notebook

 Key Insights

* Higher age is associated with increased probability of abnormal test results
* Lower physical scores may correlate with poorer hearing performance
* Machine learning models can reasonably predict test outcomes based on input features

 Future Improvements

* Add more health-related features for better prediction
* Improve model accuracy with hyperparameter tuning
* Deploy as a web application (Flask / FastAPI)
* Integrate real-time prediction system

Author

Fuad Mohammed
Machine Learning & Full-Stack Developer
