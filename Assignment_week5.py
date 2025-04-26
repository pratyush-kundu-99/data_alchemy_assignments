# Data handling
import pandas as pd  # DataFrames
import numpy as np  # Arrays

# Visualization
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical graphics
import plotly.express as px

# Utilities
import itertools  # Iterators
import math  # Math functions
from math import ceil  # Ceil function

# Dataset & outliers
from sklearn import datasets  # Datasets
from sklearn.ensemble import IsolationForest  # Anomaly detection

# UCI dataset fetching
from ucimlrepo import fetch_ucirepo  # Fetch datasets

# Statistical analysis
from scipy import stats  # Statistics
from scipy.stats import pointbiserialr  # Correlation
from scipy.stats import chi2_contingency  # Chi-squared test

# Clustering
from scipy.cluster.hierarchy import linkage  # Clustering
from scipy.spatial.distance import pdist  # Pairwise distance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Preprocessing & modeling
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.linear_model import LinearRegression  # Linear regression
from sklearn.preprocessing import OneHotEncoder  # Encoding
from sklearn.compose import ColumnTransformer  # Column transformer
from sklearn.pipeline import Pipeline  # Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree  # Decision tree (regression)
from sklearn.metrics import mean_squared_error, r2_score  # Metrics (regression)
from sklearn.linear_model import LogisticRegression  # Logistic regression
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error  # Metrics (classification, regression)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Classification metrics
from sklearn.ensemble import RandomForestRegressor  # Random forest (regression)
from sklearn.tree import DecisionTreeClassifier  # Decision tree (classification)
from sklearn.ensemble import RandomForestClassifier  # Random forest (classification)
from sklearn.model_selection import train_test_split, GridSearchCV  # Grid search
import streamlit as st

# Visualization settings
sns.set(style="whitegrid")  # Plot style

# Downloading the covid data set
covid_data = pd.read_csv("covid_data.csv")
covid_data

# Displaying the first few rows
covid_data.head() # displays the first 5 rows of the data set

# Renaming columns to snake case and lower case
covid_data.columns = covid_data.columns.str.replace(" ","_")
covid_data.columns = covid_data.columns.str.lower()
covid_data.columns

# Function to replace the date_died values with 1 for alive and 2 for dead
def replacement(x):
    if x == "9999-99-99":
        x = 1

        return x
    else:
        x = 2

        return x


data = [1,2,3,4]
replacement(3)

covid_data['date_died']= covid_data["date_died"].apply(replacement)

# Changing hipertension and date_died variable names to hypertension and status
covid_data = covid_data.rename(columns={'hipertension': 'hypertension'})
covid_data = covid_data.rename(columns={'date_died': 'status'})

# Subsetting to only have the hospitalized patients in the data set
covid_data = covid_data[covid_data['patient_type'] == 2]

# Removing the patient_type column since it only has the hospitalized records
covid_data = covid_data.drop(columns=['patient_type'])

# Subsetting the data set to include fewer variables of interest
covid_data = covid_data[['copd', 'age', 'status', 'diabetes', 'hypertension', 'clasiffication_final', 'pneumonia', 'asthma', 'cardiovascular']]

# Function to map classification
def classify(row):
    if row == 1:
        return "Mild"
    elif row == 2:
        return "Moderate"
    elif row == 3:
        return "Severe"
    else:
        return "Not a Carrier/Inconclusive"


# Function to map status
def map_status(value):
    if value == 1:
        return "Alive"
    elif value == 2:
        return "Dead"
    else:
        return "Unknown"  # For other values


# Application of the classify function to the 'clasiffication_final' column
covid_data['clasiffication_final'] = covid_data['clasiffication_final'].apply(classify)

# Application of the map_status function to the 'status' column
covid_data['status'] = covid_data['status'].apply(map_status)

# Function to map Yes/No
def map_yes_no(value):
    if value == 1:
        return "Yes"
    elif value == 2:
        return "No"
    else:
        return "Unknown"  # Applying for the other values

# List of my columns that need the change
columns_to_map = ['pneumonia', 'diabetes', 'copd', 'asthma', 
                  'hypertension','cardiovascular']

# Application of the function to change the entries in the variables from1,2 to Yes/No
for column in columns_to_map:
    if column in covid_data.columns:
        covid_data[column] = covid_data[column].apply(map_yes_no)

covid_data.head(20)

# The shape of the test data set
print(f"Dataset shape: {covid_data.shape}")

# Displaying all column names
print("\nColumn Names:") 
print(covid_data.columns.tolist())

# understanding the datatypes
covid_data.dtypes

# Checking for missing values
print("\nMissing Values Count:")
print(covid_data.isnull().sum())

# Checking out the uniques values in each column
unique_values = covid_data.apply(lambda x: x.unique())
# Display the unique values for each column
print(unique_values)

columns_to_check = ['copd', 'diabetes', 'hypertension', 'pneumonia', 'asthma', 'cardiovascular', 'status']
total_entries = covid_data.shape[0]

# Calculate counts and ratios for 'Unknown' values in the specified columns
for column in columns_to_check:
    count_unknown = (covid_data[column] == 'Unknown').sum()
    ratio_unknown = count_unknown / total_entries
    print(f"Count of Unknown in {column}: {count_unknown}")
    print(f"Ratio of Unknown in {column}: {ratio_unknown:.4f}")

# Removing the unknown values
covid_data = covid_data[~covid_data[columns_to_check].isin(['Unknown']).any(axis=1)]

# Checking out the uniques values in each column
unique_values = covid_data.apply(lambda x: x.unique())
# Display the unique values for each column
print(unique_values)

# Checking the number of 97, 98, 99 values from the age column and their ratio
count_97 = (covid_data['age'] == 97).sum()
count_98 = (covid_data['age'] == 98).sum()
count_99 = (covid_data['age'] == 99).sum()
total_entries = covid_data['age'].shape[0]
ratio_97 = count_97 / total_entries
ratio_98 = count_98 / total_entries
ratio_99 = count_99 / total_entries

print(f"Count of 97 in age: {count_97}")
print(f"Count of 98 in age: {count_98}")
print(f"Count of 99 in age: {count_99}")
print(f"Ratio of 97 in age: {ratio_97:.4f}")
print(f"Ratio of 98 in age: {ratio_98:.4f}")
print(f"Ratio of 99 in age: {ratio_99:.4f}")

# Removing the 97,98,99 that signify the missing values and unkwonn values.
covid_data = covid_data[~covid_data['age'].isin([97, 98, 99])]

# Checking the number of 97, 98, 99 values from the age column and their ratio after removing
count_97 = (covid_data['age'] == 97).sum()
count_98 = (covid_data['age'] == 98).sum()
count_99 = (covid_data['age'] == 99).sum()
total_entries = covid_data['age'].shape[0]
ratio_97 = count_97 / total_entries
ratio_98 = count_98 / total_entries
ratio_99 = count_99 / total_entries

print(f"Count of 97 in age: {count_97}")
print(f"Count of 98 in age: {count_98}")
print(f"Count of 99 in age: {count_99}")
print(f"Ratio of 97 in age: {ratio_97:.4f}")
print(f"Ratio of 98 in age: {ratio_98:.4f}")
print(f"Ratio of 99 in age: {ratio_99:.4f}")

#Summary statistics
covid_data.describe()

# Checking for outliers in age variable using a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=covid_data['age'], color='green')
plt.title('Boxplot for Age')
plt.xlabel('Age')
plt.show()

# Calculating outliers and removing them from the age variable
Q1 = covid_data['age'].quantile(0.25)
Q3 = covid_data['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = covid_data[(covid_data['age'] < lower_bound) | (covid_data['age'] > upper_bound)]
print(f"Number of outliers: {outliers.shape[0]}")
# Removing outliers from the age variable
covid_data = covid_data[(covid_data['age'] >= lower_bound) & (covid_data['age'] <= upper_bound)]

# Plotting a box plot of the age variable after removing the data outliers.
plt.figure(figsize=(8, 6))
sns.boxplot(x=covid_data['age'], color='green')
plt.title('Boxplot for Age')
plt.xlabel('Age')
plt.xlim(left=0)
plt.xticks(range(0, int(covid_data['age'].max()) + 10, 10))
plt.show()

# Creating count plots for the categorical variables
categorical_columns = covid_data.select_dtypes(include=['object', 'category']).columns
n_cols = 4
n_rows = math.ceil(len(categorical_columns) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

custom_palette = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2', '#FF69B4', '#A52A2A', '#20B2AA']

for i, column in enumerate(categorical_columns):
    sns.countplot(data=covid_data, x=column, ax=axes[i], palette=custom_palette)
    axes[i].set_title(f'Bar Plot for {column}')
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Creating box plots for categorical variables against age
categorical_columns = covid_data.select_dtypes(include=['object', 'category']).columns.tolist()
plt.figure(figsize=(15, 15))

# Looping through the categorical columns for the boxplot
for i, column in enumerate(categorical_columns):
    plt.subplot(4, 5, i + 1)  # 4 rows and 5 columns for subplot grid
    sns.boxplot(x=covid_data[column], y=covid_data['age'], palette='viridis')
    
    # Setting the title and labels of the plot
    plt.title(f'Age vs {column}')
    plt.xlabel(column)
    plt.ylabel('Age')

    plt.xticks(rotation=45, ha='right')  # Rotate x-ticks by 45 degrees, align to the right
    plt.yticks(range(0, int(covid_data['age'].max()) + 10, 10))  # Set ticks at intervals of 10

plt.tight_layout()
plt.show()


# Creating violin plots for visualization
categorical_columns = covid_data.select_dtypes(include=['object', 'category']).columns.tolist()
plt.figure(figsize=(15, 15))

# Looping through the categorical columns for the violins creation
for i, column in enumerate(categorical_columns):
    plt.subplot(4, 5, i + 1) 
    sns.violinplot(x=covid_data[column], y=covid_data['age'], palette='viridis')
    
    # Setting the plot title and labels
    plt.title(f'Age vs {column}')
    plt.xlabel(column)
    plt.ylabel('Age')

    plt.xticks(rotation=45, ha='right')  # Rotate x-ticks by 45 degrees, and align to the right
    plt.yticks(range(0, int(covid_data['age'].max()) + 10, 10))  # Code sets the ticks at intervals of 10

plt.tight_layout()
plt.show()

# Correlation matrix to visualize the relationship between the categorical values with a focus on the status against the cormobidities and age
covid_data1 = covid_data.copy()

covid_data1['status'] = covid_data1['status'].map({'Alive': 1, 'Dead': 0})
covid_data1['copd'] = covid_data1['copd'].map({'Yes': 1, 'No': 0})
covid_data1['diabetes'] = covid_data1['diabetes'].map({'Yes': 1, 'No': 0})
covid_data1['hypertension'] = covid_data1['hypertension'].map({'Yes': 1, 'No': 0})
covid_data1['pneumonia'] = covid_data1['pneumonia'].map({'Yes': 1, 'No': 0})
covid_data1['asthma'] = covid_data1['asthma'].map({'Yes': 1, 'No': 0})
covid_data1['cardiovascular'] = covid_data1['cardiovascular'].map({'Yes': 1, 'No': 0})

covid_data1 = pd.get_dummies(covid_data1, columns=['clasiffication_final'], drop_first=True)

correlation_matrix = covid_data1.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Variables Correlation Matrix')
plt.show()

# Linear regression for the dependant variable age and the independent variables.
# Creating data set copy to avoid tampering with the original data set
covid_data_copy = covid_data.copy()

# Data preprocessing
covid_data_copy['status'] = covid_data_copy['status'].map({'Alive': 1, 'Dead': 0})  # Alive=1, Dead=0
covid_data_copy['copd'] = covid_data_copy['copd'].map({'Yes': 1, 'No': 0})
covid_data_copy['diabetes'] = covid_data_copy['diabetes'].map({'Yes': 1, 'No': 0})
covid_data_copy['hypertension'] = covid_data_copy['hypertension'].map({'Yes': 1, 'No': 0})
covid_data_copy['pneumonia'] = covid_data_copy['pneumonia'].map({'Yes': 1, 'No': 0})
covid_data_copy['asthma'] = covid_data_copy['asthma'].map({'Yes': 1, 'No': 0})
covid_data_copy['cardiovascular'] = covid_data_copy['cardiovascular'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding for the 'clasiffication_final' column 
covid_data_copy = pd.get_dummies(covid_data_copy, columns=['clasiffication_final'], drop_first=True)

X = covid_data_copy.drop(columns=['age'])  # Independent variables (excluding 'age' from features)
y = covid_data_copy['age']  # Dependent variable (age)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression 
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)  # Fitting model on the training data

# Predictions
y_pred_lin = lin_reg.predict(X_test)

# Calculating performance metrics
r2_lin = r2_score(y_test, y_pred_lin)  # R-squared score
mse_lin = mean_squared_error(y_test, y_pred_lin)  # Mean Squared Error
mae_lin = mean_absolute_error(y_test, y_pred_lin)  # Mean Absolute Error
rmse_lin = np.sqrt(mse_lin)  # Root Mean Squared Error

print("Intercept:", lin_reg.intercept_)
print("Model Coefficients:")
coefficients = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

print("R-squared score on the test data:", r2_lin)

print("MSE (Mean Squared Error):", mse_lin)
print("RMSE (Root Mean Squared Error):", rmse_lin)
print("MAE (Mean Absolute Error):", mae_lin)

# Decision Tree Regressor
X = covid_data_copy.drop(columns=['age'])  
y = covid_data_copy['age']  

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_reg.predict(X_test)

r2_dt = r2_score(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)

print("Decision Tree Regressor R²:", r2_dt)
print("Decision Tree Regressor MSE:", mse_dt)

plt.figure(figsize=(20, 20))  
plot_tree(dt_reg, filled=True, feature_names=X.columns, class_names=["Age"], fontsize=10)
plt.title("Decision Tree")
plt.show()

# Random Forest Regressor
X = covid_data_copy.drop(columns=['age'])  
y = covid_data_copy['age'] 

# Splitting into training and testing sets
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor model
rf_reg = RandomForestRegressor(n_estimators=10, random_state=42)
rf_reg.fit(X_train, y_reg_train)

# Predictions
y_pred_rf = rf_reg.predict(X_test)

r2_rf = r2_score(y_reg_test, y_pred_rf)
mse_rf = mean_squared_error(y_reg_test, y_pred_rf)

print("Random Forest Regressor R²:", r2_rf)
print("Random Forest Regressor MSE:", mse_rf)


# Logistic regression to check how various cormobidities influence the likelihood of a patient being either alive or dead (status)
X = covid_data_copy.drop(columns=['status', 'age'])  
y = covid_data_copy['status']  

# Train-test split
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y, test_size=0.2, random_state=42)

# logistic regression model
log_reg = LogisticRegression(max_iter=1000)  # Set a higher max_iter in case of convergence issues
log_reg.fit(X_train, y_class_train)

# Predictions
y_pred_log = log_reg.predict(X_test)

acc_log = accuracy_score(y_class_test, y_pred_log)
print("Logistic Regression Accuracy:", acc_log)

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_pred_log)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Precision, Recall, F1-Score
precision = precision_score(y_class_test, y_pred_log)
recall = recall_score(y_class_test, y_pred_log)
f1 = f1_score(y_class_test, y_pred_log)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Classification Report 
print("\nClassification Report:")
print(classification_report(y_class_test, y_pred_log))

# Decision Tree Classifier Accuracy
X = covid_data_copy.drop(columns=['status', 'age'])  
y = covid_data_copy['status']  

# Train-test split
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier 
dt_clf = DecisionTreeClassifier(random_state=50, max_depth=5)  
dt_clf.fit(X_train, y_class_train)

# Predictions
y_pred_dt_clf = dt_clf.predict(X_test)

acc_dt_clf = accuracy_score(y_class_test, y_pred_dt_clf)
print("Decision Tree Classifier Accuracy:", acc_dt_clf)

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_pred_dt_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.show()

# Precision, Recall, F1-Score
precision = precision_score(y_class_test, y_pred_dt_clf)
recall = recall_score(y_class_test, y_pred_dt_clf)
f1 = f1_score(y_class_test, y_pred_dt_clf)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Classification Report (optional, for more detailed metrics)
print("\nClassification Report:")
print(classification_report(y_class_test, y_pred_dt_clf))


# Random Forest Classifier Accuracy
X = covid_data_copy.drop(columns=['status', 'age'])  
y = covid_data_copy['status']  

# Train-test split
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier 
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_class_train)

# Predictions
y_pred_rf_clf = rf_clf.predict(X_test)

acc_rf_clf = accuracy_score(y_class_test, y_pred_rf_clf)
print("Random Forest Classifier Accuracy:", acc_rf_clf)

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_pred_rf_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Random Forest Classifier")
plt.show()

# Precision, Recall, F1-Score
precision = precision_score(y_class_test, y_pred_rf_clf)
recall = recall_score(y_class_test, y_pred_rf_clf)
f1 = f1_score(y_class_test, y_pred_rf_clf)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Classification Report 
print("\nClassification Report:")
print(classification_report(y_class_test, y_pred_rf_clf))


# Hyperparameter Tuning
X = covid_data_copy.drop(columns=['status', 'age'])  
y = covid_data_copy['status']  

# Train-test split
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest Classifier 
rf_clf = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],  # Splitting criteria
    'max_depth': [None, 5, 10, 20],      # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],     # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],       # Minimum number of samples required to be at a leaf node
    'max_features': [None, 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}

# GridSearchCV 
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_class_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)

results_df = pd.DataFrame(grid_search.cv_results_)

print("\nGrid Search Results:")
print(results_df[['params', 'mean_test_score', 'std_test_score']])

# Train and evaluate with best parameters 
best_rf_clf = grid_search.best_estimator_

# Making predictions with the best model
y_pred_rf_clf = best_rf_clf.predict(X_test)

acc_rf_clf = accuracy_score(y_class_test, y_pred_rf_clf)
print("\nRandom Forest Classifier Accuracy with Best Parameters:", acc_rf_clf)

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_pred_rf_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Random Forest Classifier with Best Parameters")
plt.show()

# Precision, Recall, F1-Score
precision = precision_score(y_class_test, y_pred_rf_clf)
recall = recall_score(y_class_test, y_pred_rf_clf)
f1 = f1_score(y_class_test, y_pred_rf_clf)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)



# App Title
st.title("Hospitalized COVID-19 patients Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the App mode", ["Interactive EDA", "Model Hyperparameter Tuning"])

# ----------------------
# Interactive EDA Section
# ----------------------

@st.cache_data
def get_scatter_plot(x_axis, y_axis, color):
    fig = px.scatter(covid_data_copy, x=x_axis, y=y_axis, color=color,
                     title=f"{x_axis} vs {y_axis}",
                     template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def get_hist(color_col, hist_column):
    fig_hist = px.histogram(covid_data_copy, x=hist_column, color=color_col, marginal="box",
                            title=f"Distribution of {hist_column}",
                            template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

def get_box_plot(box_column):
    fig_box = px.box(covid_data_copy, x=box_column, title="Box Plot by Category")
    st.plotly_chart(fig_box, use_container_width=True)

# ----------------------
# Hyperparameter Tuning Section
# ----------------------

@st.cache_data
def hyperparameter_tuning(covid_data_copy, target_col):
    X = covid_data_copy.drop(columns=[target_col])
    y = covid_data_copy[target_col]

    # Encoding categorical target variable (if needed)
    if y.dtypes == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define RandomForestClassifier
    rf_clf = RandomForestClassifier(random_state=42)

    # Hyperparameter Grid Search
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    # GridSearchCV
    grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best Hyperparameters and Model
    best_rf_clf = grid_search.best_estimator_

    # Predictions and performance evaluation
    y_pred = best_rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return grid_search.best_params_, grid_search.best_score_, accuracy, precision, recall, f1, cm, best_rf_clf

if app_mode == "Interactive EDA":
    # Exploratory Data Analysis Section
    st.header("Exploratory Data Analysis")

    # Use preloaded covid_data_copy directly
    st.subheader("Dataset Preview")
    st.write(covid_data_copy.head())

    st.write("Dataset Dimensions:", covid_data_copy.shape)
    st.subheader("Summary Statistics")
    st.write(covid_data_copy.describe())

    # Scatter Plot
    numeric_columns = covid_data_copy.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = covid_data_copy.select_dtypes(include="object").columns.tolist()

    st.subheader("Interactive Scatter Plot")
    x_axis = st.selectbox("Select X-axis", numeric_columns, index=0)
    y_axis = st.selectbox("Select Y-axis", numeric_columns, index=1)
    color = st.selectbox("Select color grouping", categorical_columns, index=0)
    get_scatter_plot(x_axis, y_axis, color)

    # Histogram
    st.subheader("Interactive Histogram")
    hist_column = st.selectbox("Select column for histogram", numeric_columns, index=0)
    color_col = st.selectbox("Select hist grouping color", categorical_columns, index=0)
    get_hist(color_col, hist_column)

    # Box Plot
    st.subheader("Interactive Boxplot")
    box_column = st.selectbox("Select column for box plot", numeric_columns, index=0)
    get_box_plot(box_column)

    # Correlation Matrix
    st.subheader("Interactive Correlation Matrix")
    fig = px.imshow(covid_data_copy.select_dtypes(include=['number']).corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", origin='lower', title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Model Hyperparameter Tuning":
    # Hyperparameter Tuning Section
    st.header("Random Forest Hyperparameter Tuning")

    # Use preloaded covid_data_copy directly
    target_column = st.selectbox("Select Target Variable", covid_data_copy.columns, index=0)

    st.write(f"Performing hyperparameter tuning for the target column: {target_column}")

    best_params, best_score, accuracy, precision, recall, f1, cm, best_model = hyperparameter_tuning(covid_data_copy, target_column)

    st.subheader("Best Hyperparameters")
    st.write(best_params)

    st.subheader("Best Cross-Validated Accuracy")
    st.write(best_score)

    st.subheader("Model Performance on Test Data")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1-Score: {f1}")

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Best Model")
    st.pyplot()