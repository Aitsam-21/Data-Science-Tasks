Data Science Internship Tasks: Titanic EDA, Sentiment Analysis, Fraud Detection, and House Price Prediction
This repository contains four data science tasks completed as part of a Data Science Internship, demonstrating skills in exploratory data analysis, machine learning, and model implementation from scratch. The tasks are:

Titanic Dataset EDA: Exploratory analysis and visualization of the Titanic dataset.
IMDB Sentiment Analysis: Sentiment classification of IMDB movie reviews.
Credit Card Fraud Detection: Fraud detection system for credit card transactions.
House Price Prediction: Custom regression models for the Boston Housing Dataset.

Project Structure
titanic-eda/
├── titanic/
│   ├── tested.csv                    # Titanic dataset
│   ├── eda_titanic.py                # EDA script
│   ├── outliers_before.png           # Outlier visualization
│   ├── outliers_after_fare.png       # Outlier treatment visualization
│   ├── categorical_bar_charts.png    # Categorical variable plots
│   ├── numeric_histograms.png        # Numeric variable distributions
│   ├── correlation_heatmap.png       # Correlation matrix
├── sentiment_analysis/
│   ├── imdb_reviews.csv              # IMDB dataset
│   ├── sentiment_analysis.py         # Sentiment analysis script
│   ├── confusion_matrix.png          # Model performance visualization
├── fraud_detection/
│   ├── creditcard.csv                # Credit card fraud dataset
│   ├── fraud_detection.py            # Fraud detection script
│   ├── confusion_matrix.png          # Model performance visualization
├── house_price_prediction/
│   ├── house_price_prediction.py     # House price prediction script
│   ├── feature_importance.png        # Feature importance plots
├── screenshots/
│   ├── titanic_screenshot1.png       # Titanic visualization screenshot
│   ├── titanic_screenshot2.png       # Titanic visualization screenshot
│   ├── sentiment_screenshot.png      # Sentiment confusion matrix screenshot
│   ├── fraud_screenshot.png          # Fraud confusion matrix screenshot
│   ├── house_screenshot.png          # House feature importance screenshot
├── videos/
│   ├── titanic_visuals.mp4           # Titanic EDA video
│   ├── sentiment_visuals.mp4         # Sentiment analysis video
│   ├── fraud_visuals.mp4             # Fraud detection video
│   ├── house_visuals.mp4             # House price prediction video
├── README.md                         # Project documentation

Prerequisites

Python: 3.x
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, imblearn
Installation:pip install pandas numpy matplotlib seaborn scikit-learn nltk imblearn


NLTK Data (for Task 2):import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



How to Run
Task 1: Titanic Dataset EDA

Navigate to titanic/:cd titanic


Run the script:python eda_titanic.py


Outputs:
Console: Dataset shape, missing value counts, outlier stats, insights.
Visuals: outliers_before.png, outliers_after_fare.png, categorical_bar_charts.png, numeric_histograms.png, correlation_heatmap.png.



Task 2: IMDB Sentiment Analysis

Navigate to sentiment_analysis/:cd sentiment_analysis


Ensure imdb_reviews.csv is present.
Run the script:python sentiment_analysis.py


Outputs:
Console: Precision, recall, F1-score.
Visual: confusion_matrix.png.



Task 3: Credit Card Fraud Detection

Navigate to fraud_detection/:cd fraud_detection


Ensure creditcard.csv is present.
Run the script:python fraud_detection.py


Outputs:
Console: Precision, recall, F1-score, testing interface prompts.
Visual: confusion_matrix.png.



Task 4: House Price Prediction

Navigate to house_price_prediction/:cd house_price_prediction


Run the script:python house_price_prediction.py


Outputs:
Console: RMSE, R² for Linear Regression, Random Forest, XGBoost.
Visual: feature_importance.png.



Task Details
Task 1: Titanic Dataset EDA

Steps:
Loading: Loaded tested.csv using Pandas.
Cleaning:
Imputed missing Age with median, Embarked with mode.
Dropped Cabin due to high missingness.
Removed duplicates (if any).
Detected outliers in Fare and Age using IQR; capped extreme Fare values.


Visualizations:
Bar charts for Sex, Pclass, Embarked, Survived.
Histograms for Age, Fare.
Correlation heatmap for Age, Fare, Pclass, SibSp, Parch.


Insights:
Shape: ~418 rows, 12 columns.
Survival rate: ~36.36%.
More males than females; females had higher survival rates.
3rd class passengers were most common but had lower survival.
Fare and Pclass negatively correlated (higher class, higher fare).
Outliers in Fare (e.g., >$500) impacted distributions.





Task 2: IMDB Sentiment Analysis

Steps:
Preprocessing:
Loaded imdb_reviews.csv.
Tokenized reviews using NLTK.
Removed stopwords and punctuation.
Applied lemmatization for normalization.


Feature Engineering:
Converted text to TF-IDF vectors using sklearn.feature_extraction.text.TfidfVectorizer.


Model Training:
Trained Logistic Regression classifier (used sklearn as permitted).


Evaluation:
Metrics: Precision, recall, F1-score on test set.
Visualized confusion matrix.


Insights:
Balanced dataset (~50% positive, 50% negative).
F1-score: ~0.83–0.85.
Common words in positive reviews: “great,” “love”; negative: “bad,” “worst.”
TF-IDF captured sentiment effectively.





Task 3: Credit Card Fraud Detection

Steps:
Preprocessing:
Loaded creditcard.csv.
Handled imbalance (~0.17% fraud) using SMOTE (imblearn.over_sampling.SMOTE).
Normalized features (Amount, PCA components).


Model Training:
Trained Random Forest classifier (used sklearn as permitted).


Evaluation:
Metrics: Precision, recall, F1-score on test set.
Visualized confusion matrix.


Testing Interface:
Built command-line interface to input transaction features and predict fraud.


Insights:
Highly imbalanced dataset; SMOTE improved recall.
F1-score: ~0.85–0.90 for fraud class.
Key features: V1–V28 (PCA components), Amount.
Interface correctly flagged synthetic fraud cases.





Task 4: House Price Prediction

Steps:
Preprocessing:
Loaded Boston Housing Dataset via sklearn.datasets.load_boston.
Normalized 13 numerical features (CRIM, ZN, ..., LSTAT) using StandardScaler.
No categorical variables.


Model Implementation:
Custom Linear Regression (normal equations with pseudo-inverse).
Custom Random Forest (10 trees, max depth 3).
Custom XGBoost (10 estimators, learning rate 0.1, max depth 3).
Implemented from scratch without sklearn.linear_model, sklearn.ensemble, or xgboost.


Performance Comparison:
Evaluated RMSE and R² on test set (20% split).


Feature Importance:
Visualized importance (split frequency) for Random Forest and XGBoost.


Insights:
Shape: 506 rows, 14 columns (MEDV target).
Linear Regression: Best R² (0.76), RMSE (4.75).
Random Forest: R² (0.69), RMSE (5.30).
XGBoost: R² (0.71), RMSE (5.10).
Simplified trees limited tree-based performance.
Key features: RM (average rooms), LSTAT (% lower status population).





Notes

Datasets:
Ensure tested.csv, imdb_reviews.csv, creditcard.csv are in their respective folders.
Task 4 uses sklearn.datasets.load_boston(); no CSV required unless specified.


Execution:
Run scripts in their respective folders.
Task 4’s custom models are simplified (e.g., shallow trees) to meet “from scratch” requirement.


Dependencies: Install all libraries before running.
Deadline: All tasks completed and submitted by April 16, 2025.
Submission: Repository link and visuals submitted via console.

Author
Muhammad Aitsam Zulfiqar
