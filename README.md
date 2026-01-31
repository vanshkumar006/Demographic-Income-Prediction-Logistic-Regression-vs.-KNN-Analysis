                      Demographic-Income-Prediction-Logistic-Regression-vs.-KNN-Analysis

📌 Project Overview
This project focuses on predicting an individual's income level (binary: >50K or <=50K) based on a variety of demographic and socio-economic factors such as age, education, and occupation.
The core of this project is a comparative analysis between two fundamental machine learning algorithms: Logistic Regression and K-Nearest Neighbors (KNN). It features a complete pipeline including automated data cleaning, deep exploratory data analysis (EDA), and hyperparameter optimization.

🚀 Key Features
Object-Oriented Design: The entire pipeline is built within a modular Python class (IncomeClassifierProject) for scalability and clean code.
Automated Data Cleaning: Intelligent handling of real-world "messy" data, including the detection and removal of placeholder missing values (?).
Hyperparameter Tuning: A custom optimization loop to find the "Elbow Point" for the K value in KNN, ensuring the highest possible accuracy.
Advanced Visualization: Rich graphical representation of data distributions, correlations, and model error rates.

📊 Exploratory Data Analysis (EDA) Insights

Before training the models, the data revealed critical socio-economic trends:

The Age Factor: Individuals in the higher income bracket (>50k) have a significantly higher median age (~44 years) compared to the lower income group (~34 years), suggesting a strong correlation between professional experience and salary.

Education Impact: Visual analysis shows that "Bachelors", "Masters", and "Prof-school" graduates dominate the high-income category, while "HS-grad" and "Some-college" make up the bulk of the <=50k group.
Feature Correlation: The Heatmap identified age, capitalgain, and hoursperweek as the primary numeric drivers of income status.

⚙️ Technical Workflow
Preprocessing:
Cleaned the dataset from 31,978 to 30,162 valid entries.
Applied One-Hot Encoding to categorical features while dropping the first column to avoid the Dummy Variable Trap.
Target mapping: <=50k→0, >50k →1.

Model Training:
Logistic Regression: Fitted with max_iter=1000 for convergence.
KNN Optimization: Tested K values from 1 to 20, plotting misclassification counts to identify the optimal balance between bias and variance.

📈 Final Results & Comparison
The models were evaluated on a 30% hold-out test set:
Metric	Logistic Regression	K-Nearest Neighbors (K=16)
Accuracy	84.27%	84.36%
Misclassified Samples	1423	1415

Conclusion: While both models performed strongly, KNN with K=16 emerged as the winner, providing the most precise predictions for this specific demographic dataset.

