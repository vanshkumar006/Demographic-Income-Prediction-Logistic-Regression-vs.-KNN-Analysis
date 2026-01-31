import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class IncomeClassifierProject:
      """
      A unified project to analyze and predict income levels using
      Logistic Regression and K-Nearest Neighbors (KNN).
      """

      def __init__(self, filepath):
         self.filepath = filepath
         self.data = None
         self.X_train = None
         self.X_test = None
         self.y_train = None
         self.y_test = None
         self.results = {}

      def load_and_clean_data(self):
         """
         Loads data, handles ' ?' as NA, and removes missing values.
         """
         print("\n--- Step 1: Data Loading & Cleaning ---")
         try:
               # Note: The dataset uses ' ?' for missing values, we handle that here
               self.data = pd.read_csv(self.filepath, na_values=[" ?"])
               print(f"Original Data Shape: {self.data.shape}")

               # Check for nulls
               null_counts = self.data.isnull().sum()
               print(f"Missing values found:\n{null_counts[null_counts > 0]}")

               # Drop missing values
               self.data.dropna(axis=0, inplace=True)
               print(f"Shape after cleaning: {self.data.shape}")
               
               # Map Target Variable (SalStat) to 0 and 1
               # <= 50,000 becomes 0, > 50,000 becomes 1
               self.data['SalStat'] = self.data['SalStat'].map({
                  ' less than or equal to 50,000': 0,
                  ' greater than 50,000': 1
               })
               print("Target variable 'SalStat' mapped to binary (0/1).")

         except FileNotFoundError:
               print(f"Error: The file '{self.filepath}' was not found.")
               exit()

      def perform_eda(self):
         """
         Generates visualizations to understand the demographics.
         """
         print("\n--- Step 2: Exploratory Data Analysis (EDA) ---")
         
         # 1. Age vs Salary
         plt.figure(figsize=(10,6))
         sns.boxplot(x='SalStat', y='age', data=self.data, palette='coolwarm')
         plt.title('Age Distribution by Salary Status')
         plt.xlabel('Income Class (0: <=50k, 1: >50k)')
         plt.show()
         
         # 2. Education vs Salary
         plt.figure(figsize=(12,6))
         sns.countplot(y='EdType', hue='SalStat', data=self.data, palette='viridis')
         plt.title('Education Level vs Income')
         plt.show()

         # 3. Correlation Matrix (Numerical only)
         numeric_data = self.data.select_dtypes(include=[np.number])
         plt.figure(figsize=(10,8))
         sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
         plt.title('Correlation Matrix')
         plt.show()
         
         print("Visualizations generated. Proceeding to modeling...")

      def preprocess_and_split(self):
         """
         One-Hot Encodes categorical data and splits into Train/Test sets.
         """
         print("\n--- Step 3: Preprocessing & Splitting ---")
         
         # One-Hot Encoding for categorical variables
         # drop_first=True prevents multicollinearity (Dummy Variable Trap)
         data_encoded = pd.get_dummies(self.data, drop_first=True)
         
         # Separate Features (X) and Target (y)
         columns_list = list(data_encoded.columns)
         features = list(set(columns_list) - set(['SalStat']))
         
         X = data_encoded[features].values
         y = data_encoded['SalStat'].values
         
         # Split: 70% Training, 30% Testing
         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
               X, y, test_size=0.3, random_state=0
         )
         print("Data split into Training (70%) and Testing (30%) sets.")

      def run_logistic_regression(self):
         """
         Trains and evaluates the Logistic Regression model.
         """
         print("\n--- Step 4: Logistic Regression Model ---")
         
         lr_model = LogisticRegression(max_iter=1000)
         lr_model.fit(self.X_train, self.y_train)
         
         predictions = lr_model.predict(self.X_test)
         accuracy = accuracy_score(self.y_test, predictions)
         misclassified = (self.y_test != predictions).sum()
         
         print(f"Logistic Regression Accuracy: {accuracy*100:.2f}%")
         print(f"Misclassified Samples: {misclassified}")
         
         # Save results for comparison
         self.results['Logistic Regression'] = accuracy

      def run_knn(self):
         """
         Trains KNN. Also searches for the optimal 'K' value.
         """
         print("\n--- Step 5: K-Nearest Neighbors (KNN) Model ---")
         
         # 1. Search for best K (from 1 to 20)
         print("Searching for optimal K value...")
         best_k = 0
         best_acc = 0
         misclassified_list = []
         
         for k in range(1, 20):
               knn = KNeighborsClassifier(n_neighbors=k)
               knn.fit(self.X_train, self.y_train)
               pred_i = knn.predict(self.X_test)
               acc = accuracy_score(self.y_test, pred_i)
               
               misclassified_list.append((self.y_test != pred_i).sum())
               
               if acc > best_acc:
                  best_acc = acc
                  best_k = k
                  
         print(f"Optimal K found: {best_k} (Accuracy: {best_acc*100:.2f}%)")
         
         # 2. Run Final KNN with Best K
         final_knn = KNeighborsClassifier(n_neighbors=best_k)
         final_knn.fit(self.X_train, self.y_train)
         predictions = final_knn.predict(self.X_test)
         
         accuracy = accuracy_score(self.y_test, predictions)
         misclassified = (self.y_test != predictions).sum()
         
         print(f"KNN Final Accuracy: {accuracy*100:.2f}%")
         print(f"Misclassified Samples: {misclassified}")
         
         self.results['KNN'] = accuracy
         
         # Plotting the error rate for different K values
         plt.figure(figsize=(10,6))
         plt.plot(range(1, 20), misclassified_list, color='blue', linestyle='dashed', marker='o', markerfacecolor='red')
         plt.title('Misclassified Samples vs K Value')
         plt.xlabel('K Value')
         plt.ylabel('Misclassified Count')
         plt.show()

      def compare_results(self):
         """
         Prints a final comparison table.
         """
         print("\n" + "="*40)
         print(" FINAL MODEL COMPARISON ")
         print("="*40)
         
         results_df = pd.DataFrame(list(self.results.items()), columns=['Model', 'Accuracy'])
         results_df['Accuracy'] = results_df['Accuracy'] * 100
         print(results_df.sort_values(by='Accuracy', ascending=False))
         
         winner = max(self.results, key=self.results.get)
         print(f"\n>> The best performing model is: {winner}")

# --- Main Execution ---
if __name__ == "__main__":
      # Ensure 'income.csv' is in the same folder
      project = IncomeClassifierProject('income.csv')
      
      project.load_and_clean_data()
      project.perform_eda()
      project.preprocess_and_split()
      
      project.run_logistic_regression()
      project.run_knn()
      
      project.compare_results()