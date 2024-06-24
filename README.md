# Loan Approval Prediction

<img width="882" alt="Screenshot 2024-06-24 at 4 41 26 PM" src="https://github.com/JaiBhatia19/London-Bike-Rides-Dashboard/assets/143343337/77fb22fb-a327-4b0e-892e-6e8bc5b5e23a">

## Project Overview
Predicting loan approval outcomes based on applicant data using machine learning models.

## Tools and Technologies
- Python
- scikit-learn
- seaborn
- matplotlib
- imbalanced-learn

## Data
The dataset used for this project can be found here:
- [train_data.csv](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?select=train_u6lujuX_CVtuZ9i.csv)

## Notebook
The data preprocessing and modeling are performed in the following Jupyter Notebook:
- [loan_approval_prediction.ipynb](loan_approval_prediction.ipynb)

## Instructions
1. Clone the repository.
2. Open the `loan_approval_prediction.ipynb` notebook and run all cells.

## Data Preprocessing
- The dataset includes information such as loan ID, gender, marital status, dependents, education, self-employment status, applicant income, co-applicant income, loan amount, loan amount term, and credit history.
- Missing values are present in several features and are filled using appropriate imputation techniques.
- Visualizations are created to gain insights into the data, such as a count plot of the gender feature.
- Applicant income and co-applicant income features are highly correlated, so a new feature, 'total income', is created by adding them.
- Logarithmic transformation is applied to normalize the distribution of applicant income, loan amount, and loan amount term.
- Unnecessary features are dropped, and label encoding is performed on categorical features.
- Independent and dependent features are split for creating the machine learning model.

## Machine Learning Models
- Logistic regression, decision trees, random forests, and k-nearest neighbors are supervised machine learning models used for binary classification problems.
- Logistic regression is a simple classifier model that gives good accuracy (around 80%).
- Decision tree splits the data into multiple subsets based on certain conditions and then performs modeling.
- Random forest splits the data into multiple subsets and implements a decision tree model on each subset, then calculates the accuracy.
- K nearest neighbors selects the highest number of values present around a neighbor's value and predicts based on that.

## Handling Imbalanced Datasets
- Oversampling is performed using the imblearn library to generate synthetic samples for the minority class in an imbalanced dataset.
- After oversampling, the dataset is balanced, and a new logistic regression model is trained on the balanced dataset.
- The accuracy of the logistic regression model on the balanced dataset is lower compared to the imbalanced dataset, indicating that the previous model was overfitted.
- The classification report for the balanced dataset shows improved precision, recall, and F1-score values, demonstrating the effectiveness of oversampling in handling imbalanced datasets.

## Improving Model Accuracy with Resampling
- Resampling techniques, such as Decision Tree, Random Forest, and K-Nearest Neighbors, are applied to balance the data and improve model performance.
- After resampling, the classification reports showed significant improvement, with Random Forest providing the best results (90% precision and 87% accuracy).

## Findings
- Logistic Regression with resampled data provides balanced performance but with slightly lower accuracy due to addressing overfitting.
- Random Forest with resampled data provided the best accuracy and precision.
- Oversampling significantly improved the model's performance on minority class prediction, addressing class imbalance effectively.
