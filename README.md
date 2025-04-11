# Regression Model Comparison Notebook

This notebook contains an in-depth analysis and comparison of three regression models — Decision Tree, Random Forest, and XGBoost — applied to a supervised regression task.

## Project Overview

The goal of this project is to:
- Train and evaluate baseline and fine-tuned versions of popular regression models.
- Compare their performance using multiple error metrics.
- Identify the best-performing model based on real-world prediction accuracy.

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
| MSE (Mean Squared Error)  | Squared error, penalizing large mistakes |
| RMSE (Root Mean Squared Error) | Square root of MSE — interpretable in original units |
| R² (R-Squared Score)      | How much variance is explained by the model |

All metrics were computed on the test set.

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

- Random Forest (Base) gave the best overall performance, with the lowest MAE and highest R².
- Tuning helped Decision Trees but slightly hurt Random Forest and XGBoost.
- XGBoost struggled, possibly due to suboptimal tuning or overfitting.


