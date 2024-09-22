# MLDM Coursework: Machine Learning Model Comparison

**Overview**
This project explores and compares the performance of various machine learning algorithms on two datasets: real estate rental prices and socio-economic data for income classification. Seven machine learning models were implemented, each optimized and evaluated based on key metrics such as accuracy, precision, recall, F1-score, Mean Squared Error (MSE), and R-squared (R²). The goal of this project is to analyze which machine learning paradigms are most suitable for different types of tasks—classification and regression.

**Datasets**
**1. Real Estate Rental Prices (Regression Task)**
Features: Number of bathrooms, bedrooms, square footage, amenities, etc.
Target: Apartment rental price
Dataset Size: 10,000 rows
Source: UCI Machine Learning Repository
**2. Adult Income Dataset (Classification Task)**
Features: Age, education, work class, hours worked per week, etc.
Target: Binary income classification (<=50K or >50K)
Dataset Size: 48,842 instances
Source: UCI Machine Learning Repository

**Machine Learning Models**
The following machine learning algorithms were applied to both datasets:
Decision Tree
A non-linear predictive model used for both classification and regression tasks. It splits data into subsets based on feature values.
XGBoost
A gradient boosting algorithm optimized for speed and performance, particularly effective for structured data.
K-Nearest Neighbors (KNN)
An instance-based learning algorithm used for classification and regression by averaging the target values of the k-nearest neighbors.
Multi-layer Perceptron (MLP)
A type of neural network that uses backpropagation to adjust weights and improve prediction performance.
Support Vector Machine (SVM)
A classification algorithm that creates a hyperplane to separate data points between classes.
Inductive Logic Programming (ILP)
A logic-based machine learning approach using background knowledge and examples to generate hypotheses.
Deep Q-Learning (DQL)
A reinforcement learning model that combines Q-learning with deep neural networks to solve sequential decision-making tasks.

**Evaluation Metrics**
For the classification task (income prediction):

Accuracy
F1-Score
Confusion Matrix
ROC AUC
For the regression task (rental price prediction):

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R² (Coefficient of Determination)
Actual vs Predicted Plots

**Results**
Classification Task (Adult Income Dataset):
XGBoost performed the best with the highest ROC AUC score, followed by MLP and Decision Tree.
Regression Task (Real Estate Rental Prices):
XGBoost and MLP models provided the best results for price prediction, with the lowest MSE and highest R² scores.
