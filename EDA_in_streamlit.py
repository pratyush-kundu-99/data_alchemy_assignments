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
from sklearn.metrics import roc_auc_score

# Visualization settings
sns.set(style="whitegrid") 

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

## Function to map classification changes
def classify(row):
    if row == 1:
        return "Mild"
    elif row == 2:
        return "Moderate"
    elif row == 3:
        return "Severe"
    else:
        return "Not a Carrier/Inconclusive"

## Function to map status changes
def map_status(value):
    if value == 1:
        return "Alive"
    elif value == 2:
        return "Dead"
    else:
        return "Unknown"  

## Application of the classify function to the 'clasiffication_final' column
covid_data['clasiffication_final'] = covid_data['clasiffication_final'].apply(classify)

## Application of the map_status function to the 'status' column
covid_data['status'] = covid_data['status'].apply(map_status)

## Function to map Yes/No changes
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

## Application of the function to change the entries in the variables from 1,2 to Yes/No
for column in columns_to_map:
    if column in covid_data.columns:
        covid_data[column] = covid_data[column].apply(map_yes_no)

## Checking unknown values
columns_to_check = ['copd', 'diabetes', 'hypertension', 'pneumonia', 'asthma', 'cardiovascular', 'status']
total_entries = covid_data.shape[0]
## Removing the unknown values from my data set from the functions above
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

## Creating a copy for data encoding and modeling section
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

# App styling section
# Home Page of the dashboard
def main_content():
    st.markdown("""
        <h1 style="color: #FF6347; font-weight: bold;">Hospitalized COVID-19 Patient Morbidity Prediction Dashboard</h1>
    """, unsafe_allow_html=True)
    st.image("https://www.amprogress.org/wp-content/uploads/2020/03/Microbes-1.jpg")
    st.write("<p style='color: #A9A9A9;'>"
        "Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus."
        " This dashboard allows for an interactive exploration of a dataset containing information about hospitalized "
        "COVID-19 patients. The primary goal is to analyze the data in an engaging way, performing exploratory data analysis "
        "(EDA) and training machine learning model (logistic regression) to predict patient outcomes (status,main focus on whether the patient died). The dataset includes various health metrics "
        "such as age, comorbidities and the classification of the patient condition in terms of covid-19 test conducted. By analyzing these factors I aimed to "
        "better understand the key determinants of patient death outcomes in a COVID-19 hospitalization context and enhancing clinical support for COVID-19 hospitalized patients."
        "</p>", unsafe_allow_html=True) 
    st.write("<p style='color: #FF6347;'>Welcome to the Dashboard! You can use the sidebar to explore different sections.</p>", unsafe_allow_html=True)

# Problem Statement Section
def problem_statement():
    st.markdown("""
        <h2 style="color: #FF6347; font-weight: bold;">Problem Statement</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: rgba(255,99,71,0.1); padding: 20px; border-radius: 10px;">
        <h3 style="color: #FF6347;">Clinical Challenge</h3>
        <p>
        During the COVID-19 pandemic, healthcare systems faced unprecedented challenges in predicting 
        patient outcomes and allocating limited resources. This dashboard addresses the critical need 
        to understand how pre-existing conditions and demographic factors influence mortality risk 
        among hospitalized COVID-19 patients.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: rgba(255,99,71,0.1); padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3 style="color: #FF6347;">Analytical Objectives</h3>
        <ul style="list-style-type: disc; padding-left: 20px;">
            <li>Quantify the impact of comorbidities (diabetes, hypertension, COPD, etc.) on mortality risk</li>
            <li>Develop a predictive model for mortality outcomes using admission characteristics</li>
            <li>Identify high-risk patient profiles for targeted clinical interventions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: rgba(255,99,71,0.1); padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3 style="color: #FF6347;">Value Proposition</h3>
        <ul style="list-style-type: disc; padding-left: 20px;">
            <li><b>For clinicians:</b> Evidence-based risk assessment tool to support treatment decisions</li>
            <li><b>For hospital administrators:</b> Data-driven approach to resource allocation</li>
            <li><b>For public health:</b> Insights into vulnerable populations needing prioritized care</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 20px;">
        <p><b>Key Questions Addressed:</b></p>
        <ol style="padding-left: 20px;">
            <li>Which comorbidities have the strongest association with COVID-19 mortality?</li>
            <li>How does age interact with pre-existing conditions to affect outcomes?</li>
            <li>Can we reliably predict mortality risk at hospital admission?</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Interactive EDA Section
def interactive_eda():
    st.markdown("""
        <h2 style="color: #FF6347; font-weight: bold;">Interactive Exploratory Data Analysis (EDA)</h2>
    """, unsafe_allow_html=True)
    
   # Dataset Overview Table
    st.markdown("""
    <h3 style="color: #FF6347;">Dataset Overview and summary statistics</h3>
    """, unsafe_allow_html=True)
    
    overview_data = {
        "Metric": ["Total Patients", "Variables", "Mortality Rate", 
                  "Average Age", "Min Age", "Max Age"],
        "Value": [len(covid_data), len(covid_data.columns),
                 f"{(1 - covid_data_copy['status'].mean()) * 100:.1f}%",
                 f"{covid_data['age'].mean():.1f}",
                 covid_data['age'].min(),
                 covid_data['age'].max()]
    }
    st.table(pd.DataFrame(overview_data))

     # Show first 5 rows of processed data
    st.write("Processed COVID-19 Data (First 5 rows):")
    st.dataframe(covid_data.head())
    
    # Download button for processed data
    st.download_button(
        label="Download Processed Data as CSV",
        data=covid_data.to_csv(index=False).encode('utf-8'),
        file_name='covid_data.csv',
        mime='text/csv'
    )

    # Age Distribution Analysis Section
    st.markdown(f"""
    <div style="margin: 20px 0;">
        <h3 style="color: #FF6347;">Age Distribution Analysis</h3>
        <p>Age is a critical factor in COVID-19 outcomes. Our analysis reveals:</p>
        <ul>
            <li><b>Median age:</b> {covid_data['age'].median():.1f} years (IQR: {covid_data['age'].quantile(0.25):.1f}-{covid_data['age'].quantile(0.75):.1f})</li>
            <li><b>Mortality by age:</b> Patients who died were significantly older (median {covid_data[covid_data['status']=='Dead']['age'].median():.1f} years) 
            compared to survivors (median {covid_data[covid_data['status']=='Alive']['age'].median():.1f} years)</li>
            <li><b>Comorbidity age patterns:</b> Patients with pneumonia averaged {covid_data[covid_data['pneumonia']=='Yes']['age'].mean():.1f} years 
            versus {covid_data[covid_data['pneumonia']=='No']['age'].mean():.1f} years for those without</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Visualization Selection
    st.markdown("""
    <div style="margin: 20px 0;">
        <h3 style="color: #FF6347;">Interactive Age Distribution by Feature</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.radio("Select visualization type:", 
                           ["Violin Plot (shows distribution)", 
                            "Box Plot (shows quartiles)"])
    with col2:
        selected_feature = st.selectbox(
            "Select feature to analyze:",
            ['status', 'pneumonia', 'diabetes', 'hypertension', 
             'copd', 'asthma', 'cardiovascular']
        )
    
    # Visualization
    if plot_type.startswith("Violin"):
        fig = px.violin(covid_data, x=selected_feature, y='age', box=True,
                       title=f"Age Distribution by {selected_feature}",
                       color=selected_feature)
    else:
        fig = px.box(covid_data, x=selected_feature, y='age',
                    title=f"Age Distribution by {selected_feature}",
                    color=selected_feature)
    st.plotly_chart(fig, use_container_width=True)

    # Comorbidities Analysis 
    comorbidity_stats = {
        'diabetes': covid_data_copy['diabetes'].mean(),
        'hypertension': covid_data_copy['hypertension'].mean(),
        'copd': covid_data_copy['copd'].mean(),
        'asthma': covid_data_copy['asthma'].mean(),
        'cardiovascular': covid_data_copy['cardiovascular'].mean(),
        'pneumonia': covid_data_copy['pneumonia'].mean()
    }
    
    most_common = max(comorbidity_stats, key=comorbidity_stats.get)
    most_common_pct = comorbidity_stats[most_common] * 100
    
    diabetes_hypertension_dead = covid_data_copy[
        (covid_data_copy['status'] == 0) & 
        (covid_data_copy['diabetes'] == 1) & 
        (covid_data_copy['hypertension'] == 1)
    ].shape[0]
    
    total_dead = (covid_data_copy['status'] == 0).sum()
    diabetes_hypertension_pct = (diabetes_hypertension_dead / total_dead * 100) if total_dead > 0 else 0
    
    pneumonia_no_dead = (1 - covid_data_copy[covid_data_copy['pneumonia'] == 0]['status'].mean()) * 100
    pneumonia_yes_dead = (1 - covid_data_copy[covid_data_copy['pneumonia'] == 1]['status'].mean()) * 100
    
    st.markdown(f"""
    <div style="margin: 30px 0 20px 0; padding-top: 20px; border-top: 1px solid #FF6347;">
        <h3 style="color: #FF6347;">Comorbidity Analysis</h3>
        <p>Key findings about pre-existing conditions in our hospitalized patients:</p>
        <ul>
            <li><b>Most common comorbidity:</b> {most_common} at {most_common_pct:.1f}% prevalence</li>
            <li><b>Diabetes-Hypertension combo:</b> Present in {diabetes_hypertension_pct:.1f}% of fatal cases</li>
            <li><b>Pneumonia impact:</b> Mortality rate jumps from {pneumonia_no_dead:.1f}% to {pneumonia_yes_dead:.1f}% when present</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
     # Comorbidities co-occurrence visualization
    st.markdown("""
    <h4 style="color: #FF6347;">Comorbidity Co-occurrence Patterns</h4>
    <p style="color: #A9A9A9;">This heatmap shows how often conditions appear together in patients:</p>
    """, unsafe_allow_html=True)

    comorbidities = ['diabetes', 'hypertension', 'copd', 'asthma', 'cardiovascular', 'pneumonia']
    comorbidity_data = covid_data_copy[comorbidities]
    cooccurrence = comorbidity_data.T.dot(comorbidity_data)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cooccurrence, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Number of Patients with Co-occurring Conditions")
    st.pyplot(fig)
    
    # Statistical insights for clinical impact
    dh_combo_count = cooccurrence.loc['diabetes', 'hypertension']
    dh_combo_pct = (dh_combo_count / len(covid_data)) * 100
    dh_dead_pct = (covid_data_copy[
        (covid_data_copy['diabetes'] == 1) & 
        (covid_data_copy['hypertension'] == 1) & 
        (covid_data_copy['status'] == 0)
    ].shape[0] / dh_combo_count) * 100
    baseline_mortality = (1 - covid_data_copy['status'].mean()) * 100
    
    st.markdown(f"""
    <div style="background-color: rgba(255,99,71,0.1); padding: 15px; border-radius: 10px; margin: 20px 0;">
        <p><b>Clinical Insight:</b> The diabetes-hypertension combination occurs in {dh_combo_count:,} patients ({dh_combo_pct:.1f}% of cohort), 
        and {dh_dead_pct:.1f}% of these patients had fatal outcomes - significantly higher than the baseline {baseline_mortality:.1f}% mortality rate.</p>
    </div>
    """, unsafe_allow_html=True)

    # Correlation Matrix with Clinical Insights
    st.markdown("""
    <div style="margin: 30px 0; padding-top: 20px; border-top: 1px solid #FF6347;">
        <h3 style="color: #FF6347;">Demographic and commobidities correlation heat map</h3>
    </div>
    """, unsafe_allow_html=True)

    mortality_correlations = covid_data_copy.corr()['status'].sort_values(ascending=False)
    top_3_risks = mortality_correlations[1:4]  
    protective_factor = mortality_correlations.idxmin()
    
    st.markdown(f"""
    <div style="background-color: rgba(255,99,71,0.1); padding: 15px; border-radius: 10px;">
        <p><b>Key Clinical Patterns:</b></p>
        <ul>
            <li><b>Top Mortality Predictors:</b> {
                ', '.join([f"{col.replace('_',' ')} (r={val:.2f})" for col, val in top_3_risks.items()])
            }</li>
            <li><b>Potential Protective Factor:</b> {protective_factor.replace('_',' ')} (r={
                mortality_correlations.min():.2f})</li>
            <li><b>Strongest Comorbidity Pair:</b> Diabetes-Hypertension (r={
                covid_data_copy[['diabetes','hypertension']].corr().iloc[0,1]:.2f})</li>
        </ul>
        <p style="font-style: italic; color: #666;">
        Interpretation: |r| > 0.5 = Strong | 0.3-0.5 = Moderate | < 0.3 = Weak
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4 style="color: #FF6347; margin-top: 20px;">Detailed Correlation Matrix</h4>
    <p style="color: #A9A9A9;">Relationships between clinical features (excluding mortality status):</p>
    """, unsafe_allow_html=True)
    
    @st.cache_data
    def get_correlation_matrix():
        numeric_cols = covid_data_copy.select_dtypes(include=['int64','float64']).columns.drop('status')
        corr_matrix = covid_data_copy[numeric_cols].corr().round(2)
        
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale="RdBu_r",
                       title="Demographic and comobidities features correlation")
        fig.update_layout(width=900, height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    get_correlation_matrix()

    # Status outcome by Severity 
    st.markdown("""
    <div style="margin: 30px 0 20px 0; padding-top: 20px; border-top: 1px solid #FF6347;">
        <h3 style="color: #FF6347;">Outcomes by COVID-19 Severity Classification</h3>
        <p>Unexpected patterns in our severity classifications:</p>
    </div>
    """, unsafe_allow_html=True)
    
    severity_outcomes = covid_data.groupby('clasiffication_final')['status'] \
                                .value_counts(normalize=True) \
                                .unstack() \
                                .loc[['Mild','Moderate','Severe']]
    
    fig = px.bar(severity_outcomes, barmode='group',
                title="Patient Outcomes by COVID-19 Severity",
                labels={'value': 'Proportion', 'variable': 'Outcome'})
    st.plotly_chart(fig, use_container_width=True)
    
    moderate_death_pct = severity_outcomes.loc['Moderate','Dead']*100
    severe_death_pct = severity_outcomes.loc['Severe','Dead']*100
    
    st.markdown(f"""
    <div style="background-color: rgba(255,99,71,0.1); padding: 15px; border-radius: 10px;">
        <p><b>Critical Finding:</b> Moderate cases (classification 2) show a {moderate_death_pct:.1f}% mortality rate - 
        higher than severe cases (classification 3) at {severe_death_pct:.1f}%. Potential explanations include:</p>
        <ul>
            <li>More aggressive treatment protocols for severe cases</li>
            <li>Possible under-monitoring of moderate cases</li>
            <li>Classification system limitations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Modeling Section for mortality probability
def modeling():
    st.markdown("""
        <h2 style="color: #FF6347; font-weight: bold;">Modeling and Predictions</h2>
    """, unsafe_allow_html=True)
    st.write("<p style='color: #A9A9A9;'>"
        "Since the project is mainly concerned with the outcome of patient based on whether the patient had a pre existing condition like asthma, diabetes, hpertension, copd e.t.c "
        "We will train and evaluate Logistic Regression model to predict the likelihood of a patient dying if a feature is set to Yes that is 1 and output its accuracy, precision, recall and F1-score."
        "From the predictions the health care workers are able to understand the critical need to care for patients based on their pre existing conditions"
        "</p>", unsafe_allow_html=True)

# Data Source Section 
def data_source():
    st.header("Data Source", divider='red')
    
    # Creating tables for display
    tab1, tab2, tab3 = st.tabs(["Dataset Description", "Data Features", "Data Processing"])
    
    with tab1:
        st.subheader("Dataset Overview")
        st.write("The dataset contains anonymized patient records from Mexican government COVID-19 cases.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Original Dataset**")
            st.markdown("""
            - **Source:** [Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)
            - **Records:** 1,048,576 patients
            - **Features:** 21 variables
            """)
            
        with col2:
            mortality_rate = (1-covid_data_copy['status'].mean())*100
            st.info("**Processed Subset**")
            st.markdown(f"""
            - **Scope:** Hospitalized patients
            - **Records:** {len(covid_data):,} patients
            - **Mortality Rate:** {mortality_rate:.1f}%
            """)
        
        st.subheader("Data Collection Context")
        st.markdown("""
        Collected during peak COVID-19 period in Mexico:
        - Hospital admission records
        - Standardized reporting
        - Anonymized patient data
        """)
    
    with tab2:
        st.subheader("Feature Details")
        st.write("Key variables included in analysis:")
        
        # Creating feature table using native Streamlit
        features = {
            "Feature": ['status', 'age', 'diabetes', 'hypertension', 
                       'copd', 'asthma', 'cardiovascular', 
                       'clasiffication_final', 'pneumonia'],
            "Description": [
                "Patient outcome (Alive/Dead)",
                "Patient age in years",
                "Presence of diabetes (Yes/No)",
                "Presence of hypertension (Yes/No)",
                "Chronic obstructive pulmonary disease (Yes/No)",
                "Presence of asthma (Yes/No)",
                "Cardiovascular disease (Yes/No)",
                "COVID severity classification",
                "Presence of pneumonia (Yes/No)"
            ]
        }
        st.dataframe(features, hide_index=True, use_container_width=True)
        
        st.subheader("Key Predictors")
        st.markdown("""
        1. Pneumonia (strongest predictor)
        2. Age
        3. Diabetes-Hypertension combination
        """)
    
    with tab3:
        st.subheader("Data Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cleaning Steps**")
            st.markdown("""
            - Removed unknown values
            - Filtered hospitalized patients
            - Handled age outliers
            """)
            
        with col2:
            st.markdown("**Transformations**")
            st.markdown("""
            - Encoded binary variables
            - One-hot encoded classification
            - Created interaction terms
            """)
        
        outlier_pct = (outliers.shape[0]/len(covid_data))*100
        st.markdown("**Data Quality**")
        st.markdown(f"""
        - **Outliers Removed:** {outliers.shape[0]} ({outlier_pct:.1f}%)
        - **Final Sample Size:** {len(covid_data):,} patients
        """)
        
# Insights and final clinical implications section
def insights():
    st.markdown("""
        <h2 style="color: #FF6347; font-weight: bold;">Key Insights and Clinical Implications</h2>
    """, unsafe_allow_html=True)
    
    # Calculating statistics of outliers and mortality
    outlier_pct = (outliers.shape[0]/len(covid_data))*100
    mortality_rate = (1-covid_data_copy['status'].mean())*100
    
    # Data Quality Findings from the preprocessing stage
    with st.expander("🔍 Data Quality Findings", expanded=True):
        st.markdown(f"""
        - **Data Cleaning Challenges:** Removed unknown values (the missing values and unkwon values were marked by values 97,98 and 99)
        - **Outlier Management:** Identified {outliers.shape[0]} age outliers ({outlier_pct:.2f}% of records)
        - **Class Imbalance:** Mortality rate = {mortality_rate:.1f}% (required class-weighted modeling)
        """)
    
    # Associated Clinical Risk Factors
    with st.expander("⚠️ Clinical Risk Factors", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Strongest Predictors:**
            - Pneumonia (Highest odds ratio)
            - Diabetes-Hypertension combo
            - Age >60 years
            """)
            
        with col2:
            st.markdown("""
            **Unexpected Findings:**
            - Asthma showed protective effect
            - Moderate cases > Severe cases risk
            - Cardiovascular moderate impact
            """)
        
        st.markdown("""
        **Moderate vs Severe Cases Insight:**  
        *Possible explanations:*
        - More aggressive treatment for severe cases
        - Under-monitoring of moderate cases
        - Potential misclassification
        """)
    
    # The perational Applications
    with st.expander("🏥 Operational Applications", expanded=True):
        st.markdown("""
        **Clinical Decision Support:**
        - 82% recall for mortality prediction
        - Adjustable risk thresholds
        
        **Health System Planning:**
        - Identify high-risk profiles
        - ICU bed needs forecasting
        """)
    
    # Project Limitations
    with st.expander("⚠️ Limitations", expanded=True):
        st.markdown("""
        - Missing detailed lab values
        - Hospitalized-only population bias
        - From this dashboard it is clear that there is importance in the health sector looking into the patients that go home since seemingly deaths happpen at home from those that were not hospitalized
        - No treatment protocol data
        """)

# Creating sidebar with links for navigation
app_mode = st.sidebar.radio(
    "Click the radio button to view associated section", 
    ["Home", "Problem Statement", "Interactive EDA", "Modeling", "Data Source", "Insights"]
)

# Defining app mode
if app_mode == "Home":
    main_content()  
elif app_mode == "Problem Statement":
    main_content()  
    problem_statement()  
elif app_mode == "Interactive EDA":
    main_content()  
    interactive_eda()  
elif app_mode == "Modeling":
    main_content()  
    modeling()  
elif app_mode == "Data Source":
    main_content()  
    data_source()  
elif app_mode == "Insights":
    main_content()  
    insights()  

# ----------------------
# Functions for Visualizations
# ----------------------

# Correlation matrix function 
@st.cache_data
def get_correlation_matrix():
    numeric_cols = covid_data_copy.select_dtypes(include=['int64','float64']).columns
    numeric_cols = numeric_cols.drop('status')  
    
    corr_matrix = covid_data_copy[numeric_cols].corr().round(2)
    
    fig_corr = px.imshow(corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Clinical Feature Correlations (Excluding Mortality Status)")
    fig_corr.update_layout(width=900, height=700)
    st.plotly_chart(fig_corr, use_container_width=True)


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
    
# ----------------------
# Model Training Section
# ----------------------

def train_model(model_name, X_train, y_train, X_test, y_test):
    # Initialize model with key improvements
    if model_name == "Logistic Regression":
        model = LogisticRegression(
            random_state=42,
            class_weight='balanced',  # Handles class imbalance
            solver='liblinear',       # More reliable solver
            max_iter=1000            # Ensure convergence
        )
    else:
        st.error("Invalid model selected. Please choose 'Logistic Regression'.")
        return None

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 0]  # Predicting probability of death

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_proba)  
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate baseline probability (all features = No)
    baseline_vector = np.zeros((1, X_train.shape[1]))
    baseline_prob = model.predict_proba(baseline_vector)[0][0]
    
    # Individual predictions with odds ratios
    individual_predictions = {}
    features_to_predict = [col for col in X_train.columns if col not in ['age', 'status']]
    
    for feature in features_to_predict:
        feature_vector = baseline_vector.copy()
        feature_index = X_train.columns.get_loc(feature)
        feature_vector[0, feature_index] = 1  
        
        feature_prob = model.predict_proba(feature_vector)[0][0]
        odds_ratio = (feature_prob/(1-feature_prob)) / (baseline_prob/(1-baseline_prob))
        
        individual_predictions[feature] = {
            'prediction': 'Alive' if model.predict(feature_vector) == 1 else 'Deceased',
            'probability_of_death': feature_prob,
            'odds_ratio': odds_ratio,
            'risk_increase': f"{(feature_prob - baseline_prob)/baseline_prob:.1%}" if baseline_prob > 0 else "N/A"
        }

    return model, accuracy, precision, recall, f1, roc_auc, cm, individual_predictions, baseline_prob

# ----------------------
# Model Training Section
# ----------------------

if app_mode == "Modeling":
    st.markdown("""
        <h1 style="color: #FF6347; font-weight: bold;">Model Training and Evaluation</h1>
    """, unsafe_allow_html=True)

    # Cross checking combination effect sample for diabetis and hypertension since they are more common in real life
    covid_data_copy['diabetes_hypertension'] = covid_data_copy['diabetes'] * covid_data_copy['hypertension']

    # Split data into features and target (original code)
    target_column = 'status'
    X = covid_data_copy.drop(columns=[target_column])
    y = covid_data_copy[target_column]

    # Remove 'age' from the feature set 
    X = X.drop(columns=['age'])

    # Split data into train and test sets with stratification 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  
    )

    # Feature selection 
    st.markdown("""
        <h2 style="color: #FF6347; font-weight: bold;">Feature Selection</h2>
    """, unsafe_allow_html=True)
    
    selected_feature = st.selectbox(
        "Select a feature to analyze its impact on mortality:",
        [col for col in X.columns if col not in ['status']]
    )

    if st.button("Train Model and Analyze Selected Feature"):
        with st.spinner("Training model..."):
            (model, accuracy, precision, recall, f1, 
             roc_auc, cm, individual_predictions, 
             baseline_prob) = train_model(
                "Logistic Regression", 
                X_train, y_train, 
                X_test, y_test
            )

            st.markdown("""
                <h2 style="color: #FF6347; font-weight: bold;">Model Performance</h2>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
                st.metric("Recall", f"{recall:.2%}")
            with col2:
                st.metric("Precision", f"{precision:.2%}")
                st.metric("ROC AUC", f"{roc_auc:.2f}")

            # Feature impact analysis 
            st.markdown("""
                <h2 style="color: #FF6347; font-weight: bold;">Feature Impact Analysis</h2>
            """, unsafe_allow_html=True)
            
            if selected_feature in individual_predictions:
                result = individual_predictions[selected_feature]
                
                st.write(f"""
                    **When {selected_feature.replace('_', ' ')} is present:**
                    - Mortality probability: {result['probability_of_death']:.1%} 
                      (Baseline: {baseline_prob:.1%})
                    - Odds Ratio: {result['odds_ratio']:.1f}x 
                    - Risk Increase: {result['risk_increase']}
                """)
                
                # Risk indicator 
                if result['odds_ratio'] > 2:
                    st.warning("High risk factor")
                elif result['odds_ratio'] > 1.2:
                    st.info("Moderate risk factor")
                else:
                    st.success("Low risk factor")

            # Confusion matrix 
            st.markdown("""
                <h2 style="color: #FF6347; font-weight: bold;">Confusion Matrix</h2>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            ax.set_title(f"Confusion Matrix - Logistic Regression")
            st.pyplot(fig)