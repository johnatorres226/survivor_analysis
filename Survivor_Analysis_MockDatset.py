#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
import seaborn as sns

# # First Step: Let's Create the mock the Dataset

# In[4]:


#Create the dataset
# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate mock data
data = {
    "PatientID": np.arange(1, n_samples + 1),
    "Group": np.random.choice([0, 1], size=n_samples),  # 0 for control, 1 for experimental
    "Age": np.random.randint(18, 90, size=n_samples),
    "Gender": np.random.choice(["Male", "Female"], size=n_samples),
    "BMI": np.round(np.random.uniform(18.5, 40, size=n_samples), 1),
    "BloodPressure": np.round(np.random.uniform(90, 180, size=n_samples), 1),
    "Cholesterol": np.round(np.random.uniform(100, 300, size=n_samples), 1),
    "Glucose": np.round(np.random.uniform(70, 200, size=n_samples), 1),
    "SmokingStatus": np.random.choice(["Never", "Former", "Current"], size=n_samples),
    "PhysicalActivity": np.random.choice(["Low", "Moderate", "High"], size=n_samples),
    "Comorbidities": np.random.randint(0, 5, size=n_samples),
    "Medications": np.random.choice([0, 1], size=n_samples), # 0 for No Medication, 1 for Taking Some Medication
    "FollowUpTime": np.round(np.random.uniform(1, 60, size=n_samples), 1), #in months
    "Outcome": np.random.choice([0, 1], size=n_samples), # 0 for Censored, 1 for Event Occurred
    "Education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], size=n_samples),
    "RaceEthnicity": np.random.choice(["White", "Black", "Hispanic", "Asian", "Other"], size=n_samples),
    "Income": np.random.choice(["<30K", "30-60K", "60-100K", ">100K"], size=n_samples),
    "MaritalStatus": np.random.choice(["Single", "Married", "Divorced", "Widowed"], size=n_samples),
    "EmploymentStatus": np.random.choice(["Employed", "Unemployed", "Retired"], size=n_samples),
}

# Create DataFrame
mock_data = pd.DataFrame(data)
mock_data.head(10)

# # Second Step: Let's Explore the Dataset

# In[6]:


#Descriptive Stats of numerical vars
df_describe = mock_data.describe().round(2).drop(columns = ["PatientID","Group","Outcome","Medications", "Comorbidities"])
df_describe = df_describe.reset_index().drop(index=0).set_index("index")
df_describe.index.name=None
df_describe.head(10)

# In[7]:


#Demographic Information
# Age	Gender	Education	RaceEthnicity	Income	MaritalStatus	EmploymentStatus SmokingStatus	PhysicalActivity	Comorbidities

# In[8]:


# Missing Value -- General Practice, but redundant here considering this is a rand-gerenrated DF. 
missing = mock_data.isnull().sum()
print(missing)

# In[9]:


#Group Distribution
groups = mock_data.copy(deep=True)
groups['Group'] = groups['Group'].replace({0:'Experimental', 1:'Control'})
groups['Outcome'] = groups['Outcome'].replace({0:'Censored', 1:'Event Occurred'})

grp_outcome = groups.groupby(['Group', 'Outcome']).size().reset_index(name='Count')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create the bar plot
plt.figure(figsize=(12, 6), dpi=90)
bar_plot = sns.barplot(x='Group', y='Count', hue='Outcome', data=grp_outcome, palette='viridis')

# Add titles and labels
plt.title('Event Occurrence by Group Type', fontsize=18)
plt.xlabel('')
plt.ylabel('Count')
plt.legend(title='Outcome', loc="lower right")
plt.xticks(fontsize=12)

# Add count annotations to the bars
for p in bar_plot.patches:
    height = p.get_height()
    bar_plot.annotate(f'{int(height)}',
                      (p.get_x() + p.get_width() / 2., height),
                      ha='center', va='center', fontsize=14, color='black', xytext=(0, 5),
                      textcoords='offset points')

# Show the plot
plt.show()

# In[10]:


# Comparisoon of Follow-Up Time by Group

sns.boxplot(x='Group', y='FollowUpTime', data=mock_data)
plt.title('Follow-Up Time by Group')
plt.xlabel('Group (0=Control, 1=Experimental)')
plt.ylabel('Follow-Up Time (Months)')
plt.show()

# In[12]:


# Covariates
# Pairplot to examine relationships
pairplot = sns.pairplot(mock_data[['Age', 'BMI', 'Comorbidities', 'FollowUpTime', 'Outcome']], hue='Outcome')
pairplot.fig.suptitle('Pairplot of Covariates by Outcome', y=1.02)
plt.show()

# Distribution of Age by Outcome
sns.boxplot(x='Outcome', y='Age', data=mock_data)
plt.title('Age Distribution by Outcome')
plt.xlabel('Outcome (1=Event, 0=Censored)')
plt.ylabel('Age')
plt.show()

# Distribution of BMI by Outcome
sns.boxplot(x='Outcome', y='BMI', data=mock_data)
plt.title('BMI Distribution by Outcome')
plt.xlabel('Outcome (1=Event, 0=Censored)')
plt.ylabel('BMI')
plt.show()

# # Third Step: Survivor Analysis of Experimental and Control Group

# In[11]:


# Kaplan-Meier Curves
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

# Filter data for each group and check if they are empty
control_group = mock_data[mock_data['Group'] == 0]
experimental_group = mock_data[mock_data['Group'] == 1]

# Fit the data for the control group
kmf.fit(control_group['FollowUpTime'], event_observed=control_group['Outcome'], label='Control')
ax = kmf.plot_survival_function()

# Fit the data for the experimental group
kmf.fit(experimental_group['FollowUpTime'], event_observed=experimental_group['Outcome'], label='Experimental')
kmf.plot_survival_function(ax=ax)

plt.title('Kaplan-Meier Survival Curves')
plt.xlabel('Time (Months)')
plt.ylabel('Survival Probability')
plt.show()

# In[13]:


# Statistical Testing: Log-Rank Test
from lifelines.statistics import logrank_test

# Extract survival times and event indicators
control_times = control_group['FollowUpTime']
control_events = control_group['Outcome']
experimental_times = experimental_group['FollowUpTime']
experimental_events = experimental_group['Outcome']

# Perform the log-rank test
results = logrank_test(control_times, experimental_times, event_observed_A=control_events, event_observed_B=experimental_events)

# Print the results
print(results)

# In[25]:


# Cox Proportional Hazards Model
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from lifelines import CoxPHFitter

# Combine groups into a single DataFrame with the 'Group' as an additional covariate
combined_data = mock_data
combined_data['Gender'] = combined_data['Gender'].replace({'Male':1, 'Female':2})
combined_data['SmokingStatus'] = combined_data['SmokingStatus'].replace({'Never':0, 'Current':1,'Former':2})
combined_data['PhysicalActivity'] = combined_data['PhysicalActivity'].replace({'Low':1, 'Moderate':1,'High':2})
combined_data['Education'] = combined_data['Education'].replace({'High School':1, 'Bachelor':1,'Master':3, 'PhD':4})
combined_data['RaceEthnicity'] = combined_data['RaceEthnicity'].replace({'White':1, 'Black':2,'Hispanic':3, 'Asian':4, 'Other':5})
combined_data['Income'] = combined_data['Income'].replace({'<30K':1, '30-60K':2,'60-100K':3, '>100K':4})
combined_data['MaritalStatus'] = combined_data['MaritalStatus'].replace({'Single':1, 'Married':2,'Divorced':3, 'Widowed':4})
combined_data['EmploymentStatus'] = combined_data['EmploymentStatus'].replace({'Employed':1, 'Unemployed':2,'Retired':3})
combined_data['Group'] = combined_data['Group'].astype('category')

# Standardize features
X = combined_data.drop(columns=['FollowUpTime', 'Outcome'])
X = sm.add_constant(X)  # adds a constant term to the predictors
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
scaled_data = pd.DataFrame(scaled_features, columns=X.columns)
scaled_data['FollowUpTime'] = combined_data['FollowUpTime'].values
scaled_data['Outcome'] = combined_data['Outcome'].values

# Check for multicollinearity
vif = pd.DataFrame()
vif['Feature'] = scaled_data.columns
vif['VIF'] = [variance_inflation_factor(scaled_data.values, i) for i in range(scaled_data.shape[1])]
print(vif)

# In[27]:


# Check for NaNs and infinite values -- We know there are none, but good practice. 
print(scaled_data.isna().sum())
print((scaled_data == float('inf')).sum())

# In[29]:


# Initialize and fit the Cox Proportional Hazards model

#Remove variables that express multicollinearity and also those with low variance
scaled_data = scaled_data.drop(columns=['RaceEthnicity', 'Income'])
scaled_data = scaled_data.drop(columns=['const'], errors='ignore')

# Initialize CoxPHFitter with L1 regularization (Lasso)
cph = CoxPHFitter(penalizer=0.1)  # Adjust the penalizer as needed

# Fit the model
cph.fit(scaled_data, duration_col='FollowUpTime', event_col='Outcome', show_progress=True)

# Print the summary
print(cph.summary)

# In[31]:


# Model Validation
from lifelines.utils import concordance_index
from statsmodels.stats.proportion import proportion_confint

#Schoenfeld Residuals

# Fit the Cox model
cph = CoxPHFitter(penalizer=0.1)
cph.fit(scaled_data, duration_col='FollowUpTime', event_col='Outcome')

# Extract Schoenfeld residuals
residuals = cph.compute_residuals(scaled_data, kind='schoenfeld')
print(residuals.head())

# # Fourth Step: Model Validation and Sensitiviy Analysis

# In[33]:


#Log-Log Survival Plot

# Calculate survival functions for a categorical variable
for group in scaled_data['Group'].unique():
    subset = scaled_data[scaled_data['Group'] == group]
    kmf = KaplanMeierFitter()
    kmf.fit(durations=subset['FollowUpTime'], event_observed=subset['Outcome'])
    kmf.plot(label=f'Group {group}')
    
plt.xlabel('Follow-Up Time')
plt.ylabel('Survival Probability')
plt.title('Survival Curves by Group')
plt.legend()
plt.show()

# In[35]:


#C-Index

# Calculate C-index
c_index = concordance_index(scaled_data['FollowUpTime'], -cph.predict_partial_hazard(scaled_data), scaled_data['Outcome'])
print(f'C-index: {c_index}')

# In[37]:


# Residuals Analysis

residuals = cph.compute_residuals(scaled_data, kind='schoenfeld')
plt.hist(residuals)
plt.title('Histogram of Schoenfeld Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# In[39]:


# Sensitiviy Analysis

# Different Penalizer Values
penalizers = [0.01, 0.1, 1.0]
for penalizer in penalizers:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(scaled_data, duration_col='FollowUpTime', event_col='Outcome')
    print(f'Penalizer: {penalizer}')
    print(cph.summary)

# In[41]:


# Model Re-specification with Interaction Terms

scaled_data['Age_BMI'] = scaled_data['Age'] * scaled_data['BMI']
cph_interaction = CoxPHFitter(penalizer=0.1)
cph_interaction.fit(scaled_data, duration_col='FollowUpTime', event_col='Outcome')
print(cph_interaction.summary)
