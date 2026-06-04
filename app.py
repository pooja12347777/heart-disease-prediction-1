import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set page config for a premium look
st.set_page_config(
    page_title="CardioShield | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #e63946;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #d6222f;
        box-shadow: 0px 4px 10px rgba(230, 57, 70, 0.4);
        transform: translateY(-2px);
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .header-style {
        font-size: 32px;
        font-weight: 800;
        color: #1d3557;
        margin-bottom: 10px;
    }
    .subheader-style {
        font-size: 18px;
        color: #457b9d;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Caching Data Loading and Model Training
# ==========================================
@st.cache_resource
def load_and_train_model():
    # Load dataset
    df = pd.read_csv('heart_1.csv')
    
    # Split features and target
    X = df.drop(columns='target')
    Y = df['target']
    
    # Train test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)
    
    # Get feature names to construct input dataframes later
    feature_names = X.columns.tolist()
    
    return model, feature_names

# Load model and features
with st.spinner("Initializing predictive medical model..."):
    rf_model, feature_columns = load_and_train_model()

# Sidebar content
st.sidebar.markdown("<h1 style='text-align: center; font-size: 80px; margin-bottom: 0;'>❤️</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='text-align: center; margin-top: 0; color: #1d3557;'>CardioShield AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    "CardioShield uses a state-of-the-art Random Forest machine learning classifier "
    "trained on clinical cardiology data to estimate the likelihood of coronary artery disease."
)
st.sidebar.info(
    "⚠️ **Disclaimer:** This tool is for informational/educational purposes only and "
    "does not constitute professional medical advice, diagnosis, or treatment."
)

# App header
st.markdown("<div class='header-style'>❤️ Heart Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-style'>Enter the clinical parameters below to assess heart health risk.</div>", unsafe_allow_html=True)

# Main form split into columns
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Patient Demographics")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
    
    sex_label = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0
    
    cp_label = st.selectbox(
        "Chest Pain Type (cp)",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )
    # Mapping cp labels to original numeric codes
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_map[cp_label]

with col2:
    st.markdown("### 🩺 Clinical Measurements")
    trestbps = st.number_input("Resting Blood Pressure (trestbps) in mm Hg", min_value=50, max_value=250, value=120, step=1)
    chol = st.number_input("Serum Cholesterol (chol) in mg/dl", min_value=100, max_value=600, value=200, step=1)
    
    fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["False", "True"])
    fbs = 1 if fbs_label == "True" else 0
    
    restecg_label = st.selectbox(
        "Resting Electrocardiographic Results (restecg)",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )
    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_map[restecg_label]

with col3:
    st.markdown("### ⚡ Exercise & Heart Performance")
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150, step=1)
    
    exang_label = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
    exang = 1 if exang_label == "Yes" else 0
    
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    
    slope_label = st.selectbox(
        "Slope of the Peak Exercise ST Segment (slope)",
        ["Upsloping", "Flat", "Downsloping"]
    )
    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope = slope_map[slope_label]
    
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
    
    thal_label = st.selectbox(
        "Thalassemia (thal)",
        ["Normal", "Fixed Defect", "Reversible Defect", "Unspecified/Other"]
    )
    thal_map = {
        "Unspecified/Other": 0,
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }
    thal = thal_map[thal_label]

st.markdown("</div>", unsafe_allow_html=True)

# Prediction execution section
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Analyze Cardiac Risk Profile")

if predict_btn:
    # Prepare input feature values as a dict matching exactly the columns used in model.fit()
    input_dict = {
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    }
    
    # Convert to DataFrame to match training headers and avoid Scikit-Learn warnings
    input_df = pd.DataFrame(input_dict)
    
    # Perform prediction
    prediction = rf_model.predict(input_df)
    probabilities = rf_model.predict_proba(input_df)[0]
    risk_percentage = round(probabilities[1] * 100, 2)
    
    st.markdown("---")
    st.markdown("### 📊 Diagnostic Analysis Summary")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if prediction[0] == 0:
            st.success(
                f"### Low Risk Identified (Confidence: {round(probabilities[0] * 100, 2)}%)\n\n"
                "The analysis suggests that the patient has a **low likelihood** of having coronary heart disease based on the provided symptoms and clinical tests."
            )
        else:
            st.error(
                f"### High Risk Identified (Confidence: {round(probabilities[1] * 100, 2)}%)\n\n"
                "The analysis suggests that the patient has a **high likelihood** of coronary heart disease. It is strongly recommended to consult a cardiologist for further diagnostic screening."
            )
            
    with res_col2:
        st.metric(
            label="Coronary Artery Disease Probability",
            value=f"{risk_percentage}%",
            delta="High Risk" if prediction[0] == 1 else "Normal Range",
            delta_color="inverse"
        )
