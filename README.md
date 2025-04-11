# Bangalore Real Estate Price Prediction

This notebook contains an in-depth analysis and comparison of three regression models — Decision Tree, Random Forest, and XGBoost — applied to predict property prices in Bangalore based on various features.

## Project Overview

The goal of this project is to:

- Predict property prices in Bangalore using historical real estate data.
- Train and evaluate both baseline and fine-tuned versions of popular regression models.
- Compare their performance using industry-standard error metrics.
- Identify the most reliable model for accurate property price estimation.

## Models Trained

| Model Variant         | Description |
|-----------------------|-------------|
| Decision Tree (Base)  | Default hyperparameters |
| Decision Tree (Tuned) | Manually tuned for max depth, min samples, etc. |
| Random Forest (Base)  | Ensemble of trees with default settings |
| Random Forest (Tuned) | Tuned for number of estimators, max features, etc. |
| XGBoost (Base)        | Gradient boosting model with defaults |
| XGBoost (Tuned)       | Tuned for learning rate, depth, gamma, etc. |

## Evaluation Metrics Used

| Metric | Meaning |
|--------|---------|
| MAE (Mean Absolute Error) | Average error in predictions |
| MSE (Mean Squared Error)  | Squared error, penalizing larger mistakes |
| RMSE (Root Mean Squared Error) | Square root of MSE, in same units as target (price) |
| R² (R-Squared Score)      | Proportion of variance in price explained by the model |

All metrics were computed on the test set using real Bangalore housing data.

## Results Summary

| Model               | MAE | MSE | RMSE | R² | Verdict |
|--------------------|------|------|------|------|---------|
| Decision Tree (Base) | 1.55 | 36.74 | 6.06 | 0.9918 | Decent |
| Decision Tree (Tuned) | 1.52 | 35.81 | 5.98 | 0.9942 | Improved |
| Random Forest (Base) | 0.83 | 25.68 | 5.07 | 0.9942 | Best Overall |
| Random Forest (Tuned) | 0.80 | 34.24 | 5.85 | 0.9923 | Slight Drop |
| XGBoost (Base) | 2.15 | 48.90 | 6.99 | 0.9890 | Weak |
| XGBoost (Tuned) | 2.42 | 53.37 | 5.98 | 0.9880 | Tuning Worsened It |

## Key Insights

- **Random Forest (Base)** achieved the best overall performance, with the lowest MAE and highest R².
- **Tuning improved** Decision Tree performance slightly but **did not benefit XGBoost**, which may have been prone to overfitting or improperly tuned.
- **XGBoost underperformed**, suggesting it may not be ideal for this specific dataset or requires more careful hyperparameter tuning.


