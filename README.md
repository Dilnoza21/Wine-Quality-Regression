# Wine Quality Prediction — Regression Project

## Overview

This project predicts the quality score of red wine (on a scale from 3 to 8) based on its physicochemical properties, using multiple regression models. The goal is to determine how accurately measurable chemical attributes — such as alcohol content, acidity, and sulphates — can predict a wine's quality rating.

## Dataset

- **Source:** Red Wine Quality dataset (`winequality-red.csv`)
- **Rows:** 1,599 wine samples
- **Features:** 11 physicochemical properties (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- **Target:** `quality` — an integer score from 3 to 8, originally assigned by wine tasters
- **Missing values:** None
- Google Colab link: https://colab.research.google.com/drive/1Ug7GBxv8BEHJ4uZ6cXrhe3EnP1-IbErv?usp=sharing 

## Exploratory Data Analysis

- Distribution of quality scores shows most wines cluster around scores 5 and 6, with very few samples at the extremes (3, 4, 8).
- Boxplots of each feature grouped by quality reveal which properties shift most as quality changes.
- A correlation heatmap shows that **alcohol** (positive) and **volatile acidity** (negative) have the strongest relationships with wine quality.

## Preprocessing

- Features were standardized using `StandardScaler`.
- Data was split into training (80%) and test (20%) sets.
- Unlike a classification approach, the quality score was **kept as a continuous variable** (not binned into categories), since the task is regression.

## Models Trained

| Model | Description |
|---|---|
| K-Nearest Neighbors Regressor (k=3, k=5) | Predicts quality by averaging the scores of the nearest similar wines |
| Random Forest Regressor | Ensemble of decision trees, averaging their predictions |
| Decision Tree Regressor | Single tree-based model |
| SGD Regressor | Linear model fit using stochastic gradient descent |

Each model was evaluated using:
- **MSE (Mean Squared Error)** — average squared prediction error (lower is better)
- **MAE (Mean Absolute Error)** — average absolute prediction error (lower is better)
- **R² Score** — proportion of variance explained by the model (higher is better, max 1.0)
- **10-fold Cross-Validation** — to check consistency of performance across different data splits

## Results

| Model | R² Score | MSE |
|---|---|---|
| KNN (k=3) | 0.303 | 0.456 |
| KNN (k=5) | 0.329 | 0.439 |
| **Random Forest** | **0.527** | **0.309** |
| Decision Tree | 0.101 | 0.588 |
| SGD | 0.403 | 0.390 |

## Conclusion

**Random Forest Regressor** performed best, explaining about 53% of the variance in wine quality with the lowest error among all models tested. Its ensemble approach — combining many decision trees — allows it to capture non-linear relationships between chemical properties and quality that simpler models miss.

The **Decision Tree Regressor** performed worst, likely due to overfitting: a single tree memorizes patterns in the training data rather than learning generalizable rules, so it fails to predict well on unseen wines.

An R² of 0.53 is a moderate result rather than an exceptional one. This is expected — wine quality ratings involve human subjectivity (taste, aroma, personal preference) that chemical measurements alone cannot fully capture.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn, plotly express

## How to Run

1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn plotly`
2. Place `winequality-red.csv` in the working directory
3. Run the notebook cells in order
