# Data handling
import pandas as pd  # DataFrames
import numpy as np  # Arrays

# Visualization
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical graphics
import plotly.express as px
import plotly.express as px
import plotly.graph_objects as go

# Utilities
import math  # Math functions
from math import ceil  # Ceil function

# Statistical analysis
from scipy import stats  # Statistics

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

# Data preprocessing
## Renaming columns to snake case and lower case
covid_data.columns = covid_data.columns.str.replace(" ", "_")
covid_data.columns = covid_data.columns.str.lower()

## Function to replace the date_died values with 1 for alive and 2 for dead
def replacement(x):
    if x == "9999-99-99":
        return 1
    else:
        return 2

covid_data['date_died'] = covid_data["date_died"].apply(replacement)

## Changing hipertension and date_died variable names to hypertension and status
covid_data = covid_data.rename(columns={'hipertension': 'hypertension'})
covid_data = covid_data.rename(columns={'date_died': 'status'})

## Subsetting to only have the hospitalized patients in the data set
covid_data = covid_data[covid_data['patient_type'] == 2]

## Removing the patient_type column since it only has the hospitalized records
covid_data = covid_data.drop(columns=['patient_type'])

## Subsetting the data set to include fewer variables of interest
covid_data = covid_data[['copd', 'age', 'status', 'diabetes', 'hypertension', 'clasiffication_final', 'pneumonia', 'asthma', 'cardiovascular']]

## Function to map classification
def classify(row):
    if row == 1:
        return "Mild"
    elif row == 2:
        return "Moderate"
    elif row == 3:
        return "Severe"
    else:
        return "Not a Carrier/Inconclusive"

## Function to map status
def map_status(value):
    if value == 1:
        return "Alive"
    elif value == 2:
        return "Dead"
    else:
        return "Unknown"  # For other values

## Application of the classify function to the 'clasiffication_final' column
covid_data['clasiffication_final'] = covid_data['clasiffication_final'].apply(classify)

## Application of the map_status function to the 'status' column
covid_data['status'] = covid_data['status'].apply(map_status)

## Function to map Yes/No
def map_yes_no(value):
    if value == 1:
        return "Yes"
    elif value == 2:
        return "No"
    else:
        return "Unknown"  

## List of my columns that need the change
columns_to_map = ['pneumonia', 'diabetes', 'copd', 'asthma', 
                  'hypertension','cardiovascular']

## Application of the function to change the entries in the variables from1,2 to Yes/No
for column in columns_to_map:
    if column in covid_data.columns:
        covid_data[column] = covid_data[column].apply(map_yes_no)

## Unknown values
columns_to_check = ['copd', 'diabetes', 'hypertension', 'pneumonia', 'asthma', 'cardiovascular', 'status']
total_entries = covid_data.shape[0]
## Removing the unknown values
covid_data = covid_data[~covid_data[columns_to_check].isin(['Unknown']).any(axis=1)]

## Removing the 97,98,99 that signify the missing values and unknown values.
covid_data = covid_data[~covid_data['age'].isin([97, 98, 99])]

## Calculating outliers and removing them from the age variable
Q1 = covid_data['age'].quantile(0.25)
Q3 = covid_data['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = covid_data[(covid_data['age'] < lower_bound) | (covid_data['age'] > upper_bound)]
print(f"Number of outliers: {outliers.shape[0]}")
## Removing outliers from the age variable
covid_data = covid_data[(covid_data['age'] >= lower_bound) & (covid_data['age'] <= upper_bound)]

## Creating a copy for data encoding
covid_data_copy = covid_data.copy()

## Data preprocessing for the binary categorical data
covid_data_copy['status'] = covid_data_copy['status'].map({'Alive': 1, 'Dead': 0})  
covid_data_copy['copd'] = covid_data_copy['copd'].map({'Yes': 1, 'No': 0})
covid_data_copy['diabetes'] = covid_data_copy['diabetes'].map({'Yes': 1, 'No': 0})
covid_data_copy['hypertension'] = covid_data_copy['hypertension'].map({'Yes': 1, 'No': 0})
covid_data_copy['pneumonia'] = covid_data_copy['pneumonia'].map({'Yes': 1, 'No': 0})
covid_data_copy['asthma'] = covid_data_copy['asthma'].map({'Yes': 1, 'No': 0})
covid_data_copy['cardiovascular'] = covid_data_copy['cardiovascular'].map({'Yes': 1, 'No': 0})

## One-Hot Encoding for the 'clasiffication_final' column 
covid_data_copy = pd.get_dummies(covid_data_copy, columns=['clasiffication_final'], drop_first=True)

# Application set up
# Setting an application title
st.markdown("""
    <h1 style="color: #FF6347; font-weight: bold;">Hospitalized COVID-19 Patients Dashboard</h1>
""", unsafe_allow_html=True)

# Display a headline image (using the provided URL)
st.image("https://www.amprogress.org/wp-content/uploads/2020/03/Microbes-1.jpg", use_container_width=True)

# About project and data set
st.markdown("""
    <p style="font-size: 18px; color: #808080;">
        Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.
        This dashboard allows for an interactive exploration of a dataset containing information about hospitalized 
        COVID-19 patients. The primary goal is to analyze the data in an engaging way, performing exploratory data analysis 
        (EDA) and training machine learning models to predict patient outcomes (status, Alive or Dead). The dataset includes various health metrics 
        such as age, comorbidities and the classification of the patient condition in terms of covid-19 test conducted. By analyzing these factors we aim to 
        better understand the key determinants of patient outcomes in a COVID-19 hospitalization context and likelihood of survival following certain conditions.
    </p>
    <p style="font-size: 16px; color: #808080;">
        The hospitalized covid-19 patients dashboard project includes the following functionalities:
        <ul>
            <li><b>Exploratory Data Analysis (EDA):</b> Visualize and explore the dataset with interactive plots including 
                violin plots, box plots and correlation matrices.</li>
            <li><b>Model Training:</b> Train and evaluate machine learning models to predict patient survival and classification.</li>
        </ul>
            <p style="font-size: 16px; color: #808080;">
        Below is a brief description of the columns in the subset dataset:
    </p>
    <table style="width: 100%; border: 1px solid #ccc; border-collapse: collapse; font-size: 14px;">
        <thead>
            <tr style="background-color: #FF6347; color: white;">
                <th style="padding: 8px; text-align: left;">Column Name</th>
                <th style="padding: 8px; text-align: left;">Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 8px; border-top: 1px solid #ccc;">age</td>
                <td style="padding: 8px; border-top: 1px solid #ccc;">The age of the patient.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">copd</td>
                <td style="padding: 8px;">Chronic Obstructive Pulmonary Disease (COPD) condition of the patient (Yes/No).</td>
            </tr>
            <tr>
                <td style="padding: 8px;">diabetes</td>
                <td style="padding: 8px;">Presence of diabetes in the patient (Yes/No).</td>
            </tr>
            <tr>
                <td style="padding: 8px;">hypertension</td>
                <td style="padding: 8px;">Presence of hypertension in the patient (Yes/No).</td>
            </tr>
            <tr>
                <td style="padding: 8px;">pneumonia</td>
                <td style="padding: 8px;">Presence of pneumonia in the patient (Yes/No).</td>
            </tr>
            <tr>
                <td style="padding: 8px;">asthma</td>
                <td style="padding: 8px;">Presence of asthma in the patient (Yes/No).</td>
            </tr>
            <tr>
                <td style="padding: 8px;">cardiovascular</td>
                <td style="padding: 8px;">Presence of cardiovascular disease in the patient (Yes/No).</td>
            </tr>
            <tr>
                <td style="padding: 8px;">status</td>
                <td style="padding: 8px;">Status of the patient after hospitalization (Alive/Dead).</td>
            </tr>
            <tr>
                <td style="padding: 8px;">clasiffication_final</td>
                <td style="padding: 8px;">Final classification of the patient’s condition (Mild/Moderate/Severe/Not a Carrier/Inconclusive).</td>
            </tr>
        </tbody>
    </table>
    
    <h2 style="color: #FF6347; font-weight: bold;">Data Source</h2>
    <p style="font-size: 16px; color: #808080;">
        The dataset was downloaded from Kaggle website (https://www.kaggle.com/datasets/meirnizri/covid19-dataset?resource=download&select=Covid+Data.csv) provided by the Mexican government. 
        This dataset contains an enormous number of anonymized patient-related information including pre-conditions. 
        The raw dataset consists of 21 unique features and 1,048,576 unique patients.
        The subset used here is of patient type 2 i.e. the hospitalized patients only with only 9 variables picked after cleaning with records of 187,056 patients for the analysis and model training.
    </p>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the App mode", ["Interactive EDA", "Model Training"])

# Displaying variable names of covid_data_copy
st.sidebar.subheader("Variable Names in Dataset")
st.sidebar.write(covid_data_copy.columns.tolist())

# Clickable links
st.sidebar.markdown("""
    [Exploratory Data Analysis](#exploratory-data-analysis)
    [Model Training](#model-training)
""", unsafe_allow_html=True)

# ----------------------
# Functions for Visualizations
# ----------------------

# Function to replace ticks on plots (status, classification, and Yes/No)
def update_ticks(fig, column):
    if column == 'status':
        # Update ticks for status (1 = Alive, 0 = Dead)
        fig.update_layout(
            xaxis=dict(
                tickvals=[1, 2],
                ticktext=["Alive ", "Dead "]
            )
        )
    elif column == 'clasiffication_final':
        # Update ticks for classification (1 = Mild, 2 = Moderate, 3 = Severe, 4+ = Not a Carrier/Inconclusive)
        fig.update_layout(
            xaxis=dict(
                tickvals=[1, 2, 3, 4],
                ticktext=["Mild ", "Moderate ", "Severe ", "Not a Carrier/Inconclusive "]
            )
        )
    else:
        # For Yes/No columns, mapping 1 -> Yes, 0 -> No
        fig.update_layout(
            xaxis=dict(
                tickvals=[1, 0],
                ticktext=["Yes ", "No "]
            )
        )
    return fig


# Function to generate violin plot with tick updates
def get_violin_plot(categorical_column):
    fig_violin = px.violin(covid_data, x=categorical_column, y='age', box=True, 
                           points="all", 
                           title=f"Age Distribution by {categorical_column}",
                           template="plotly_dark")
    
    fig_violin = update_ticks(fig_violin, categorical_column)
    st.plotly_chart(fig_violin, use_container_width=True)

# Function to generate box plot with tick updates
def get_box_plot(categorical_column):
    fig_box = px.box(covid_data, x=categorical_column, y='age', 
                     title=f"Age Distribution by {categorical_column}",
                     template="plotly_dark", 
                     boxmode="group")
    
    fig_box = update_ticks(fig_box, categorical_column)
    st.plotly_chart(fig_box, use_container_width=True)
    

 # Correlation matrix function
@st.cache_data
def get_correlation_matrix():
    all_columns = covid_data_copy.columns.tolist()
    
    # Calculate the correlation matrix 
    corr_matrix = covid_data_copy[all_columns].corr()

    # Rounding the correlation matrix values to 2 decimal places
    corr_matrix = corr_matrix.round(2)
    corr_df = pd.DataFrame(corr_matrix, columns=all_columns, index=all_columns)

    # Plot the correlation matrix 
    fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", 
                         color_continuous_scale="RdBu_r", origin="lower", 
                         title="Correlation Matrix",
                         labels=dict(x="Variables", y="Variables", color="Correlation"))
    
    # Setting correlation matrix layout
    fig_corr.update_layout(
        autosize=True,  
        width=1000,  
        height=1000,  
        xaxis_tickangle=-45,  
        yaxis_tickangle=45,  
        title_x=0.5,  
        margin=dict(t=40, b=40, l=40, r=40)  
    )
    
    fig_corr.update_traces(
        textfont=dict(size=14, color='black', family="Arial, sans-serif", weight='bold')  
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------
# Model Training Section
# ----------------------

def train_model(model_name, X_train, y_train, X_test, y_test):
    """
    Train a selected model and return performance metrics.
    """
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        st.error("Invalid model selected.")
        return None

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, precision, recall, f1, cm

# ----------------------
# Interactive EDA Section
# ----------------------

if app_mode == "Interactive EDA":
    st.markdown("""
        <h1 style="color: #FF6347; font-weight: bold;">Exploratory Data Analysis</h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h2 style="color: #FF6347; font-weight: bold;">Dataset Preview</h2>
    """, unsafe_allow_html=True)
    st.write(covid_data.head())  

    st.write("Dataset Dimensions:", covid_data.shape)
    
    st.markdown("""
        <h2 style="color: #FF6347; font-weight: bold;">Summary Statistics</h2>
    """, unsafe_allow_html=True)
    st.write(covid_data.describe())

    # Dropdowns for variable selection visualization
    st.markdown("""
        <h2 style="color: #FF6347; font-weight: bold;">Visualizations for Categorical Variables</h2>
    """, unsafe_allow_html=True)

    # List of categorical columns
    categorical_columns = ['copd', 'diabetes', 'hypertension', 'pneumonia', 'asthma', 'cardiovascular', 'status', 'clasiffication_final']
    categorical_columns = [col for col in categorical_columns if col in covid_data.columns]


    # Dropdown for Violin Plot
    st.markdown("""
        <h3 style="color: #FF6347; font-weight: bold;">Violin Plot</h3>
    """, unsafe_allow_html=True)
    violin_column = st.selectbox("Select a Categorical Variable for Violin Plot", categorical_columns)
    if violin_column:
        get_violin_plot(categorical_column=violin_column)

    # Dropdown for Box Plot
    st.markdown("""
        <h3 style="color: #FF6347; font-weight: bold;">Box Plot</h3>
    """, unsafe_allow_html=True)
    box_column = st.selectbox("Select a Categorical Variable for Box Plot", categorical_columns)
    if box_column:
        get_box_plot(categorical_column=box_column)

    # Correlation Matrix 
    st.markdown("""
        <h3 style="color: #FF6347; font-weight: bold;">Correlation matrix for the variables</h3>
    """, unsafe_allow_html=True)
    get_correlation_matrix()

# ----------------------
# Model Training Section
# ----------------------

elif app_mode == "Model Training":
    st.markdown("""
        <h1 style="color: #FF6347; font-weight: bold;">Model Training and Evaluation</h1>
    """, unsafe_allow_html=True)

    # Select target variable
    target_column = st.selectbox("Select Target Variable", covid_data_copy.columns, index=0)

    # Split data into features and target
    X = covid_data_copy.drop(columns=[target_column])
    y = covid_data_copy[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    model_name = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Decision Tree"])

    # Train and evaluate model
    if st.button("Train Model"):
        st.write(f"Training {model_name}...")
        model, accuracy, precision, recall, f1, cm = train_model(model_name, X_train, y_train, X_test, y_test)

        # Display performance metrics
        st.markdown("""
            <h2 style="color: #FF6347; font-weight: bold;">Model Performance</h2>
        """, unsafe_allow_html=True)
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1-Score: {f1:.4f}")

        # Display confusion matrix
        st.markdown("""
            <h2 style="color: #FF6347; font-weight: bold;">Confusion Matrix</h2>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(f"Confusion Matrix - {model_name}")
        st.pyplot(fig)