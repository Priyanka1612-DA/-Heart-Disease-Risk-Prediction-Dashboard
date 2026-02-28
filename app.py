import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------- Custom Styling --------------------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
p{
color:black;
}
.stMetric {
    background-color: white;
    color: black;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load & Train Model --------------------
df = pd.read_csv("heart_disease.csv")
df.fillna(0, inplace=True)

X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------- Header --------------------
st.title("❤️ Heart Disease Risk Prediction Dashboard")
st.markdown("### AI-Based Cardiovascular Risk Assessment System")
st.markdown("---")

# -------------------- Sidebar --------------------
st.sidebar.header("📝 Patient Information")

male = st.sidebar.checkbox("Male")
age = st.sidebar.slider("Age", 18, 100, 50)
education = st.sidebar.slider("Education Level (0-4)", 0, 4, 2)
currentSmoker = st.sidebar.checkbox("Current Smoker")
cigsPerDay = st.sidebar.slider("Cigarettes Per Day", 0, 70, 0)
BPMeds = st.sidebar.checkbox("On BP Medication")
prevalentStroke = st.sidebar.checkbox("Prevalent Stroke")
prevalentHyp = st.sidebar.checkbox("Prevalent Hypertension")
diabetes = st.sidebar.checkbox("Diabetes")
totChol = st.sidebar.slider("Total Cholesterol", 100, 700, 200)
sysBP = st.sidebar.slider("Systolic BP", 80.0, 300.0, 120.0)
diaBP = st.sidebar.slider("Diastolic BP", 40.0, 180.0, 80.0)
BMI = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
heartRate = st.sidebar.slider("Heart Rate", 40, 120, 70)
glucose = st.sidebar.slider("Glucose", 40, 400, 80)

# -------------------- Prediction --------------------
if st.sidebar.button("🔍 Predict Risk"):

    input_data = np.array([[int(male), age, education, int(currentSmoker),
                            cigsPerDay, int(BPMeds), int(prevalentStroke),
                            int(prevalentHyp), int(diabetes), totChol,
                            sysBP, diaBP, BMI, heartRate, glucose]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # -------------------- KPI Section --------------------
    st.markdown("## 🏥 Risk Assessment Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🧠 Model Accuracy", f"{round(accuracy*100,2)}%")

    with col2:
        st.metric("❤️ Risk Probability", f"{round(probability*100,2)}%")

    with col3:
        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        st.metric("⚕️ Risk Level", risk_label)

    st.markdown("---")

    # -------------------- Risk Indicator --------------------
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease Detected")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.progress(int(probability * 100))

    st.markdown(" ")

    # -------------------- Tabs Section --------------------
    tab1, tab2, tab3 = st.tabs(
        ["📊 Vital Statistics", "🧪 Health Indicators", "📄 Patient Summary"]
    )

    # -------- Tab 1 --------
    with tab1:
        colA, colB = st.columns(2)

        with colA:
            st.subheader("Blood Pressure Analysis")
            bp_df = pd.DataFrame({
                "Type": ["Systolic BP", "Diastolic BP"],
                "Value": [sysBP, diaBP]
            })
            st.bar_chart(bp_df.set_index("Type"))

        with colB:
            st.subheader("Heart & Body Metrics")
            hb_df = pd.DataFrame({
                "Metric": ["Heart Rate", "BMI"],
                "Value": [heartRate, BMI]
            })
            st.bar_chart(hb_df.set_index("Metric"))

    # -------- Tab 2 --------
    with tab2:
        st.subheader("Cholesterol & Glucose Analysis")

        health_df = pd.DataFrame({
            "Indicator": ["Cholesterol", "Glucose", "Cigarettes/Day"],
            "Value": [totChol, glucose, cigsPerDay]
        })
        st.bar_chart(health_df.set_index("Indicator"))

    # -------- Tab 3 --------
    with tab3:
        summary = pd.DataFrame({
            "Feature": X.columns,
            "Value": input_data.flatten()
        })
        st.dataframe(summary, use_container_width=True)

else:
    st.info("👈 Enter patient details in the sidebar and click Predict Risk.")
    st.markdown("### 👩‍💻 Developed by Priyanka kumari | Data Science Portfolio Project")


