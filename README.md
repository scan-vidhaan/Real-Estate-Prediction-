
# Bangalore Property Price Prediction  
This project is a part of the AAI-XXX course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Project Status**: Active

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bangalore-property-prediction.git
   cd bangalore-property-prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   Open `training.ipynb` using Jupyter Notebook or any IDE like VS Code.

---

## Project Intro/Objective
The main objective of this project is to develop a machine learning-based regression system that predicts property prices in Bangalore, India. With the rapid urbanization and growth in the real estate market, having a tool that can estimate housing costs will benefit buyers, sellers, and real estate consultants.

This system enables informed decisions by analyzing factors like area, location, amenities, and other influential variables. We aim to build, tune, and compare different models (Decision Tree, Random Forest, XGBoost) to identify the best-performing approach for accurate predictions.

---

## Partner(s)/Contributor(s)
- Vidhaan Appaji
- Antereep Chakraborty
- Ankur Bhagat


---

## Methods Used
- Machine Learning  
- Data Visualization  
- Data Cleaning & Preprocessing  
- Model Evaluation & Comparison  
- Hyperparameter Tuning  
- Regression Techniques

---

## Technologies
- Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost)  
- Jupyter Notebook  
- Git & GitHub

---

## Project Description

### Dataset
- Source: Kaggle – Bangalore House Price dataset  
- Size: ~13,000 rows, 7+ key variables  
- Target Variable: `price` (in Lakhs INR)

### Data Preprocessing
- Removed outliers  
- Dropped rows with missing values  
- Encoded categorical features  
- Normalized numerical features

### Key Questions
- Can we accurately predict house prices based on property features?  
- Which regression model performs best for this prediction task?  
- What are the effects of hyperparameter tuning on performance?

### Models Compared
1. Decision Tree Regressor  
2. Random Forest Regressor  
3. XGBoost Regressor

Each model was trained with both:
- Base/default parameters  
- Tuned parameters using GridSearchCV or manual tuning

### Performance Metrics
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score

All evaluation results are stored and compared in a `model_metrics_comparison.csv` file. Visualization includes comparative bar plots for each metric.

### Challenges
- Handling missing and inconsistent data  
- Overfitting in complex models  
- Time-consuming grid search for hyperparameter tuning

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
Special thanks to Professor Ankur Bhist for guidance and feedback throughout the project.
