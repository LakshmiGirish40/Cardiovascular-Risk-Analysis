# Cardiovascular-Risk-Analysis
**Project Title:**
**Cardiac Health Detective (or Cardiovascular Risk Explorer)**
**Objective:**
- The primary goal of this project is to build an interactive web application that helps in predicting the likelihood of **heart disease based on various health factors.**
 - The app allows users to input medical data such as age, cholesterol levels, blood pressure, and more, and uses a pre-trained machine learning model to classify 
  the risk of heart disease. The project aims to:
 - Educate users on cardiovascular risk factors.
 - Assist in preliminary heart disease risk assessments.
 - Provide real-time predictions using machine learning algorithms.

**Project Overview:**
  - This project employs a machine learning model to analyze a dataset of patient health metrics and predict heart disease. The app is built using Streamlit for an 
  interactive user interface and a RandomForest Classifier to make predictions. Users can adjust health parameters using sliders and buttons, and the app will 
  display the result of the prediction along with some educational information about the significance of each parameter.

**Technologies Used:**
  - **Python:** Programming language for developing the model and backend logic.
  - **Streamlit:** Web framework used for building the interactive web app.
  - **RandomForest Classifier:** Machine learning algorithm used for classification.
  - **Pandas & Scikit-learn:** Libraries for data manipulation, preprocessing, and model training.
  - **HTML & CSS:** To customize the look and feel of the application.
  - **Jupyter Notebook:** For initial exploratory data analysis and model building.

**Dataset:**
- The heart disease dataset and its significance in predicting heart disease:

  - **1. Age:**
    **Description:** The age of the individual.
    **Significance:** Age is a major risk factor for heart disease. The risk increases with age, especially for men after 45 and women after 55.
  - **2. Sex:**
     - **Description:** Gender of the individual.
     - **0:** Female
     - **1:** Male
     - **Significance:** Men are at higher risk of heart disease than women, though the risk for women increases after menopause.
  - **3. Chest Pain Type (cp):**
      - **Description:** Type of chest pain experienced by the individual.
      - **0:** Typical angina (chest pain related to decreased blood supply to the heart)
      - **1:** Atypical angina
      - **2:** Non-anginal pain
      - **3:** Asymptomatic
      - **Significance:** Chest pain type is a strong indicator of heart disease. Typical angina is closely related to coronary artery blockages, making it a key 
     feature in the prediction.
  - **4. Resting Blood Pressure (trestbps):**
      - **Description:** Resting blood pressure in mmHg.
      - **Significance:** High resting blood pressure (hypertension) can damage arteries over time, making them more prone to heart disease. Generally, a systolic 
           blood pressure over 140 mmHg indicates an increased risk
  - **5. Serum Cholesterol (chol):**
      - **Description:** Serum cholesterol in mg/dL.
      - **Significance:** High cholesterol levels can lead to the formation of plaques in the arteries, which can block blood flow to the heart.
      - **LDL (Low-Density Lipoprotein):** "Bad" cholesterol, which increases the risk of plaque formation.
      - **HDL (High-Density Lipoprotein):** "Good" cholesterol, which helps clear cholesterol from arteries. Total cholesterol over 200 mg/dL increases the 
                  risk of heart disease.
  - **6. Fasting Blood Sugar (fbs):**
      - **Description:** Fasting blood sugar level (whether blood sugar is greater than 120 mg/dL).
        - **0:** Fasting blood sugar ≤ 120 mg/dL
        - **1:** Fasting blood sugar > 120 mg/dL
        - **Significance:** Elevated fasting blood sugar levels may indicate diabetes, which is a major risk factor for heart disease. Diabetics are more 
                               prone to develop cardiovascular conditions.
  - **7. Resting ECG Results (restecg):**
    - **Description:** Results of the resting electrocardiogram.
       - **0:** Normal
       - **1:** ST-T wave abnormality (signs of a heart attack or ischemia)
       - **2:** Showing probable or definite left ventricular hypertrophy (thickening of the heart's main pumping chamber)
       - **Significance:** Abnormal ECG results, such as ST-T wave abnormalities or left ventricular hypertrophy, are signs of heart strain or damage, indicating a 
           higher likelihood of heart disease.
  - **8. Maximum Heart Rate Achieved (thalach):**
      - **Description:** Maximum heart rate achieved during a stress test.
      - **Significance:** Lower maximum heart rates may indicate reduced cardiac function. The heart's inability to reach a high heart rate can signal potential 
         heart problems or blockages.
  - **9. Exercise-Induced Angina (exang):**
      - **Description:** Whether angina (chest pain) is induced by exercise.
      - **0:** No
      - **1:** Yes
      - **Significance:** Exercise-induced angina is a strong indicator of poor blood flow to the heart, which often correlates with blocked or narrowed arteries.
  - **10. ST Depression Induced by Exercise (oldpeak):**
      - **Description:** ST segment depression induced by exercise, measured in mm.
      - **Significance:** ST depression on an electrocardiogram indicates the presence of ischemia, which occurs when there’s insufficient blood flow to the heart 
            muscle. The higher the value, the more severe the ischemia, indicating a higher risk for heart disease.
  - **11. Slope of Peak Exercise ST Segment (slope):**
    - **Description:** The slope of the peak exercise ST segment.
    - **1:** Upsloping (less risk)
    - **2:** Flat (moderate risk)
    - **3:** Downsloping (higher risk)
    - **Significance:** The ST segment slope during a stress test provides information on how well the heart copes with exercise. A downsloping ST segment is 
          usually a sign of coronary artery disease.
  - **12. Number of Major Vessels Colored by Fluoroscopy (ca):**
     - **Description:** Number of major blood vessels (ranging from 0 to 3) that are colored by fluoroscopy during an angiogram.
     - **Significance:** The more vessels affected, the higher the risk of heart disease. A higher number (closer to 3) indicates more widespread coronary artery 
      disease.
  - **13. Thalassemia (thal):**
     - **Description:** A blood disorder related to hemoglobin.
     - **1:** Normal
     - **2:** Fixed defect (permanent damage to heart muscle)
     - **3:** Reversible defect (damage that can improve with treatment)
     - **Significance:** Thalassemia can affect oxygen transport in the blood, and certain types (especially fixed or reversible defects) are associated with 
        increased heart disease risk.
  - **14. Target (Heart Disease):**
     - **Description:** The target variable (outcome) indicating the presence of heart disease.
     - **0:** No heart disease
     - **1:** Heart disease
     - **Significance:** This is the outcome the model is trying to predict, whether the individual has heart disease based on the other variables.



























Explore in Streamlit App: https://cardiovascular-risk-analysis.streamlit.app/
