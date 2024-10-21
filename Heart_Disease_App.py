#streamlit run Breast_Cancer_Analysis_App.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
#Streamlit

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.impute import SimpleImputer  # Import the imputer

# Dummy dataset loading (replace with actual data)
data = pd.read_csv(r"heart_disease_dataset.csv")

# Separate features and target
X = data.drop('target', axis=1)
y = data['target'] 

# Handle missing values using SimpleImputer
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#=======================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
print("Random forest hyper parameter tuning best attributes and values")
rf_base1 = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=4, min_samples_leaf=1,                                 min_samples_split=2, n_estimators=100, random_state=0)
rf_base1.fit(X_train, y_train)
y_pred_rf1 = rf_base1.predict(X_test)
print(classification_report(y_train, rf_base1.predict(X_train)))
# Evaluate the optimized model on the test data
print(classification_report(y_test, rf_base1.predict(X_test)))
randoforest_acc1 = accuracy_score(y_test,y_pred_rf1)
#==============================================================================
# Interactive user input via sliders for prediction
st.write("<h1 style='text-align: left; color: purple;'>Cardiovascular Risk Analysis</h1>", unsafe_allow_html=True)



# Sliders with detailed descriptions
input_data = {
    'age': [st.slider('Age', min_value=1, max_value=100, value=25, step=1, help='Age of the individual')],
    'sex': [st.slider('Sex', min_value=0, max_value=1, value=0, step=1, help='0 for Female, 1 for Male')],
    'cp': [st.slider('Chest Pain Type (cp)', min_value=0, max_value=3, value=1, step=1, help='0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic')],
    'trestbps': [st.slider('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120, step=5, help='Resting blood pressure in mmHg')],
    'chol': [st.slider('Cholesterol (chol)', min_value=120, max_value=400, value=200, step=5, help='Serum cholesterol in mg/dL')],
    'fbs': [st.slider('Fasting Blood Sugar (fbs)', min_value=60, max_value=300, value=0, step=1, help='1 if fasting blood sugar > Normal: Less than 100 mg/dL,Prediabetes: 100 to 125 mg/dL, 0 otherwise')],
    'restecg': [st.slider('Resting ECG (restecg)', min_value=0, max_value=3, value=0, step=1, help='0: Normal, 1: ST-T wave abnormality, 2: Probable/definite left ventricular hypertrophy')],
    'thalach': [st.slider('Max Heart Rate Achieved (thalach)', min_value=70, max_value=210, value=140, step=5, help='Maximum heart rate achieved')],
    'exang': [st.slider('Exercise Induced Angina (exang)', min_value=0, max_value=1, value=0, step=1, help='1 if exercise-induced angina, 0 otherwise')],
    'oldpeak': [st.slider('ST Depression (oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1, help='ST depression induced by exercise relative to rest')],
    'slope': [st.slider('Slope of Peak Exercise ST Segment (slope)', min_value=1, max_value=3, value=2, step=1, help='1: Upsloping, 2: Flat, 3: Downsloping')],
    'ca': [st.slider('Number of Major Vessels (ca)', min_value=0, max_value=3, value=0, step=1, help='Number of major vessels colored by fluoroscopy (0-3)')],
    'thal': [st.slider('Thalassemia (thal)', min_value=1, max_value=3, value=2, step=1, help='1: Normal, 2: Fixed defect, 3: Reversible defect')],
}


#Convert the input into a DataFrame for prediction
input_data_df = pd.DataFrame(input_data)

# Ensure the DataFrame has all expected features
input_data_df1 = input_data_df.reindex(columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'], fill_value=0)


# Scale the input data using the same scaler as used in training
input_data_scaled = scaler.transform(input_data_df1)

# Make the prediction using the trained RandomForest model
if st.button("Predict"):
    try:
        prediction = rf_base1.predict(input_data_scaled)  # Use the RandomForest model here
        st.write(f"The predicted class is: {' the Patient have Heart Disease problem' if prediction[0] == 1 else 'No Heart Disease'}")
        
        if prediction[0] == 1:
            st.write("<h4 style='text-align: left; color: red;'> 1 : The person is likely to have heart disease.</h4>", unsafe_allow_html=True)
        else:
            st.write("<h4 style='text-align: left; color: green;'> 0 : The person is unlikely to have heart disease.</h4>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    

#====================================================================================================
    
import streamlit as st

# Description
st.sidebar.write("""
                 
## Cardiovascular Risk Analysis Detailed Descriptions

### Age
Risk increases with age.

### Sex
Men are at higher risk, but risk increases for women after menopause.
- **0**: Male
- **1** : Female

### Chest Pain Type (cp)
- **0**: Typical angina (related to decreased blood supply to the heart)
- **1**: Atypical angina
- **2**: Non-anginal pain
- **3**: Asymptomatic

### Resting Blood Pressure (trestbps)
High levels (typically above 140/90 mmHg) suggest a risk.

### Cholesterol (chol)
- Levels above 200 mg/dL increase risk.
- **LDL (bad cholesterol)** above 100 mg/dL
- **HDL (good cholesterol)** below 40 mg/dL

### Fasting Blood Sugar (fbs)
- **Normal:** Less than 100 mg/dL
- **Prediabetes:** 100 to 125 mg/dL
- **Diabetes:** 126 mg/dL or higher
Levels above 120 mg/dL may indicate diabetes, which increases risk.

### Resting ECG (restecg)
- **0**: Normal
- **1**: ST-T wave abnormality (T wave inversions and/or ST elevation or depression)
- **2**: Showing probable or definite left ventricular hypertrophy

### Max Heart Rate Achieved (thalach)
Lower max heart rate can indicate issues; typically, below 150 bpm can be a concern.

### Exercise Induced Angina (exang)
- **1**: Angina induced during exercise
- **0**: No angina

### ST Depression Induced by Exercise (oldpeak)
Higher values indicate higher risk (e.g., values >1.0).

### Slope of Peak Exercise ST Segment (slope)
- **1**: Upsloping (less risk)
- **2**: Flat (moderate risk)
- **3**: Downsloping (higher risk)

### Number of Major Vessels Colored by Fluoroscopy (ca)
Higher numbers of colored vessels indicate higher risk.

### Thalassemia (thal)
- **1**: Normal
- **2**: Fixed defect (more risk)
- **3**: Reversible defect (even more risk)
""") 


#============================================================================
# Add custom CSS to increase the size of the sliders and their font
import base64
st.markdown("""
<style>
    
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {{
        background: url("data:"h2.jpg";base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: repeat;
        color: white;
    }}
    .stButton button {
       font-size: 28px;  /* Adjust the font size as needed */
       background-color: #2A71A4; /* Green background */
       color: white; /* White text */
       border: none;
       padding: 15px 32px;
       text-align: center;
       text-decoration: none;
       display: inline-block;
       font-size: 16px;
       margin: 4px 2px;
       cursor: pointer;
   }
   .stSlider > div {
        width: 700px;  /* Adjust the width as needed */
    }
    .stSlider label {
        font-size: 34px;  /* Adjust the font size as needed */
    }
    .stSlider [role="slider"] {
        font-size: 24px;  /* Adjust the slider content font size as needed */
    }
    .stSlider span {
        font-size: 26px;  /* Adjust the slider attribute font size as needed */
    }
    
</style>
""", unsafe_allow_html=True)
