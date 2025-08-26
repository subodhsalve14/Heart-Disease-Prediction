import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# --- Caching the Model ---
# This decorator ensures that the data is loaded and the model is trained only once.
@st.cache_resource
def load_and_train_model():
    """
    Loads the heart disease dataset, cleans it, and trains a 
    Random Forest Classifier.
    """
    # Load the data from the CSV file
    data = pd.read_csv('heart.csv')
    
    # Remove duplicates as done in the notebook
    data = data.drop_duplicates()
    
    # Define features (X) and target (y)
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Initialize and train the Random Forest model
    # This uses the same model that gave the best accuracy in your notebook.
    model = RandomForestClassifier()
    model.fit(X, y)
    
    return model

# Load the trained model
model = load_and_train_model()

# --- User Interface ---
st.title("❤️ Heart Disease Prediction App")
st.write(
    "This app uses a Random Forest model to predict the likelihood of heart disease. "
    "Please enter the patient's details in the sidebar to get a prediction."
)

st.sidebar.header("Patient's Medical Data")

# --- Function to get user inputs from the sidebar ---
def get_user_input():
    """
    Creates sidebar widgets to get input from the user.
    Returns a pandas DataFrame with the user's inputs.
    """
    # Dictionary mappings for select boxes
    sex_map = {'Female': 0, 'Male': 1}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'False (< 120 mg/dl)': 0, 'True (> 120 mg/dl)': 1}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'No': 0, 'Yes': 1}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Normal': 0, 'Fixed defect': 1, 'Reversible defect': 2, 'Severe defect': 3}

    # Creating widgets
    age = st.sidebar.slider('Age', 20, 100, 50)
    sex_label = st.sidebar.selectbox('Sex', list(sex_map.keys()))
    cp_label = st.sidebar.selectbox('Chest Pain Type (cp)', list(cp_map.keys()))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 90, 200, 125)
    chol = st.sidebar.slider('Serum Cholestoral (chol)', 120, 600, 212)
    fbs_label = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', list(fbs_map.keys()))
    restecg_label = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', list(restecg_map.keys()))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 70, 220, 168)
    exang_label = st.sidebar.selectbox('Exercise Induced Angina (exang)', list(exang_map.keys()))
    oldpeak = st.sidebar.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0)
    slope_label = st.sidebar.selectbox('Slope of the peak exercise ST segment', list(slope_map.keys()))
    ca = st.sidebar.selectbox('Number of major vessels colored by flourosopy (ca)', [0, 1, 2, 3, 4])
    thal_label = st.sidebar.selectbox('Thalassemia (thal)', list(thal_map.keys()))

    # Map text labels to numerical values
    sex = sex_map[sex_label]
    cp = cp_map[cp_label]
    fbs = fbs_map[fbs_label]
    restecg = restecg_map[restecg_label]
    exang = exang_map[exang_label]
    slope = slope_map[slope_label]
    thal = thal_map[thal_label]

    # Create a dictionary for the user's data
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Convert dictionary to a pandas DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input_df = get_user_input()

# --- Main Panel for Displaying Input and Prediction ---
st.subheader("Patient's Input Data")
st.dataframe(user_input_df)

# Prediction Button
if st.button('Get Prediction', type="primary", use_container_width=True):
    # Make prediction
    prediction = model.predict(user_input_df)
    prediction_proba = model.predict_proba(user_input_df)

    st.subheader('Prediction Result')
    
    # Display the result
    if prediction[0] == 0:
        st.success('**Conclusion: The model predicts that the patient does NOT have heart disease.**', icon="✅")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
    else:
        st.error('**Conclusion: The model predicts that the patient HAS heart disease.**', icon="⚠️")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")