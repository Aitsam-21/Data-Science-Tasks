# Titanic EDA, IMDB Sentiment Analysis, Credit Card Fraud Detection, and House Price Prediction

This repository contains four data science tasks:
1. **Titanic Dataset EDA**: Exploratory analysis of the Titanic test dataset.
2. **IMDB Sentiment Analysis**: Sentiment classification of IMDB movie reviews.
3. **Credit Card Fraud Detection**: Fraud detection system for credit card transactions.
4. **House Price Prediction**: Regression models for a house price dataset.

## Project Structure
- `titanic/`
  - `tested.csv`: Titanic dataset.
  - `eda_titanic.py`: EDA script.
  - `outliers_before.png`, `outliers_after_fare.png`, `categorical_bar_charts.png`, `numeric_histograms.png`, `correlation_heatmap.png`: Visualizations.
- `sentiment_analysis/`
  - `imdb_reviews.csv`: IMDB dataset.
  - `sentiment_analysis.py`: Sentiment analysis script.
  - `confusion_matrix.png`: Confusion matrix.
- `fraud_detection/`
  - `creditcard.csv`: Fraud dataset.
  - `fraud_detection.py`: Fraud detection script.
  - `confusion_matrix.png`: Confusion matrix.
- `house_price_prediction/`
  - `housing.csv`: House price dataset.
  - `house_price_prediction.py`: Prediction script.
  - `feature_importance.png`: Feature importance plots.
- `screenshots/`: Visualization screenshots.
  - `titanic_screenshot*.png`: Titanic plots.
  - `sentiment_screenshot.png`: Sentiment confusion matrix.
  - `fraud_screenshot.png`: Fraud confusion matrix.
  - `house_screenshot.png`: House feature importance.
- `videos/`: Screen recordings.
  - `titanic_visuals.mp4`: Titanic EDA.
  - `sentiment_visuals.mp4`: Sentiment analysis.
  - `fraud_visuals.mp4`: Fraud detection.
  - `house_visuals.mp4`: House price prediction.
- `README.md`: Project overview.

## Prerequisites
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `imblearn`
- Install: `pip install pandas numpy matplotlib seaborn scikit-learn nltk imblearn`
- NLTK data: `nltk.download('punkt')`, `nltk.download('stopwords')`, `nltk.download('wordnet')`

## How to Run

### Titanic EDA
1. Navigate to `titanic/`.
2. Run: `python eda_titanic.py`
3. Outputs: Console info, PNGs in `titanic/`.

### IMDB Sentiment Analysis
1. Navigate to `sentiment_analysis/`.
2. Ensure `imdb_reviews.csv` is present.
3. Run: `python sentiment_analysis.py`
4. Outputs: Console metrics, `confusion_matrix.png`.

### Credit Card Fraud Detection
1. Navigate to `fraud_detection/`.
2. Ensure `creditcard.csv` is present.
3. Run: `python fraud_detection.py`
4. Outputs: Console metrics, `confusion_matrix.png`, interface.

### House Price Prediction
1. Navigate to `house_price_prediction/`.
2. Ensure `housing.csv` is present.
3. Run: `python house_price_prediction.py`
4. Outputs: Console metrics, `feature_importance.png`.

## Titanic EDA Details
- **Steps**: Loaded `tested.csv`, cleaned data, visualized patterns.
- **Observations**: 418 rows, ~36.36% survival, more males, 3rd class majority.

## IMDB Sentiment Analysis Details
- **Steps**: Loaded `imdb.csv`, preprocessed text, trained Logistic Regression.
- **Observations**: F1-score ~0.83-0.85.

## Credit Card Fraud Detection Details
- **Steps**: Loaded `creditcard.csv`, applied SMOTE, trained Random Forest, built interface.
- **Observations**: F1-score ~0.85-0.90 for fraud.

## House Price Prediction Details
- **Steps**:
  - Loaded `housing.csv`.
  - Preprocessed: Normalized numerical features, encoded categorical.
  - Implemented custom Linear Regression, Random Forest, XGBoost.
  - Compared RMSE, R².
  - Visualized feature importance.
- **Observations**:
  - Linear Regression: Best R² (e.g., ~0.78).
  - Random Forest/XGBoost: Slightly lower R² (e.g., ~0.75) due to simplified trees.
  - Key features: Likely size-related (e.g., `sqft_living`) and location.


## Notes
- Ensure datasets (`tested.csv`, `imdb.csv`, `creditcard.csv`, `housing.csv`) are in respective folders.
- Adjust `house_price_prediction.py` target column if not 'price'.
- Deadline: April 16, 2025.

## Author
MUHAMMAD AITSAM ZULFIQAR
