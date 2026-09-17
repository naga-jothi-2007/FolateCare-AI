import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FolateCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "extended_random_forest.pkl"
TRAIN_PATH = "extended_train.csv"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "RIDAGEYR",
    "BMXBMI",
    "LBXHGB",
    "LBXHCT",
    "LBXRBCSI",
    "DSQTFDFE",
    "DSQTFA",
    "DSQTVB12",
    "DSQTVB6",
    "DSQTIRON"
]

FEATURE_LABELS = {
    "RIDAGEYR": "Age",
    "BMXBMI": "BMI",
    "LBXHGB": "Hemoglobin",
    "LBXHCT": "Hematocrit",
    "LBXRBCSI": "RBC Count",
    "DSQTFDFE": "Dietary Folate",
    "DSQTFA": "Dietary Folic Acid",
    "DSQTVB12": "Vitamin B12",
    "DSQTVB6": "Vitamin B6",
    "DSQTIRON": "Dietary Iron"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    return pd.read_csv(TRAIN_PATH)


model = load_model()
train_df = load_training_data()

MEDIANS = train_df[FEATURES].median()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GENERAL PAGE
   ========================================================== */

.stApp {
    background: #f5f9f7;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #e8f5ef 0%,
        #eef5ff 100%
    );
    border-right: 1px solid #d5e5df;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #174d3c;
}

section[data-testid="stSidebar"] label {
    color: #34534a;
    font-weight: 600;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    background: linear-gradient(
        135deg,
        #dff7eb 0%,
        #e4f1ff 55%,
        #f0e8ff 100%
    );

    border: 1px solid #d0e8dc;
    border-radius: 28px;
    padding: 42px 45px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(31, 70, 55, 0.08);
}

.hero-title {
    font-size: 44px;
    font-weight: 850;
    color: #124b3a;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 21px;
    font-weight: 650;
    color: #315f52;
    line-height: 1.5;
    margin-bottom: 12px;
}

.hero-description {
    font-size: 16px;
    color: #58746c;
    margin-bottom: 22px;
}

.hero-badge {
    display: inline-block;
    background: #ffffff;
    color: #14734f;
    padding: 10px 18px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 750;
    border: 1px solid #c6e3d5;
}


/* ==========================================================
   INFORMATION BOX
   ========================================================== */

.info-box {
    background: #edf6ff;
    border: 1px solid #d2e5f7;
    border-left: 5px solid #4d8edb;
    border-radius: 12px;
    padding: 16px 20px;
    color: #39566d;
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 25px;
}


/* ==========================================================
   SECTION TITLE
   ========================================================== */

.section-title {
    font-size: 27px;
    font-weight: 800;
    color: #173f66;
    margin-top: 25px;
    margin-bottom: 17px;
}


/* ==========================================================
   FEATURE CARDS
   ========================================================== */

.feature-card {
    background: #ffffff;
    border: 1px solid #dce7e3;
    border-radius: 18px;
    padding: 24px;
    min-height: 175px;
    box-shadow: 0 6px 18px rgba(40, 70, 60, 0.06);
    transition: 0.2s;
}

.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(40, 70, 60, 0.10);
}

.feature-icon {
    font-size: 34px;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 18px;
    font-weight: 800;
    color: #173f66;
    margin-bottom: 8px;
}

.feature-text {
    font-size: 13px;
    color: #657783;
    line-height: 1.6;
}


/* ==========================================================
   PREDICTION BUTTON
   ========================================================== */

div.stButton > button {
    background: linear-gradient(
        90deg,
        #16865f,
        #237db5
    );

    color: white;
    border: none;
    border-radius: 12px;
    padding: 13px 20px;
    font-size: 17px;
    font-weight: 750;
    min-height: 50px;
    box-shadow: 0 6px 16px rgba(25, 110, 85, 0.20);
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #11704e,
        #17658f
    );
    color: white;
}


/* ==========================================================
   RISK RESULT
   ========================================================== */

.high-risk {
    background: linear-gradient(
        135deg,
        #fff1f2,
        #fff7f7
    );

    border: 2px solid #f5b7bd;
    border-radius: 22px;
    padding: 34px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(170, 50, 60, 0.08);
}

.low-risk {
    background: linear-gradient(
        135deg,
        #ecfdf3,
        #f6fff9
    );

    border: 2px solid #a9dfbf;
    border-radius: 22px;
    padding: 34px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(30, 120, 75, 0.08);
}

.risk-heading {
    font-size: 35px;
    font-weight: 850;
    color: #263238;
    margin-bottom: 12px;
}

.risk-probability {
    font-size: 24px;
    font-weight: 750;
    color: #35515a;
    margin-bottom: 10px;
}

.risk-description {
    font-size: 14px;
    color: #5d6d73;
    line-height: 1.6;
}


/* ==========================================================
   METRIC CARDS
   ========================================================== */

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce7e3;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(40, 70, 60, 0.05);
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;
    padding: 35px 10px 10px 10px;
    color: #718096;
    font-size: 13px;
    line-height: 1.7;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border: none;
    border-top: 1px solid #dbe5e1;
    margin: 30px 0;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌿 FolateCare AI")
    st.caption("Pregnant Woman Risk Assessment")

    st.markdown("---")

    st.markdown("### 👩 Maternal Information")

    age = st.number_input(
        "Age (years)",
        min_value=18.0,
        max_value=50.0,
        value=30.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=10.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

    st.markdown("### 🩸 Hematological Information")

    hgb = st.number_input(
        "Hemoglobin (g/dL)",
        min_value=5.0,
        max_value=20.0,
        value=12.0,
        step=0.1
    )

    hct = st.number_input(
        "Hematocrit (%)",
        min_value=15.0,
        max_value=55.0,
        value=36.0,
        step=0.1
    )

    rbc = st.number_input(
        "RBC Count",
        min_value=2.0,
        max_value=7.0,
        value=4.2,
        step=0.01
    )

    st.markdown("### 🥗 Nutritional Information")

    dietary_folate = st.number_input(
        "Dietary Folate",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )

    dietary_folic_acid = st.number_input(
        "Dietary Folic Acid",
        min_value=0.0,
        max_value=10000.0,
        value=600.0,
        step=10.0
    )

    vitamin_b12 = st.number_input(
        "Vitamin B12",
        min_value=0.0,
        max_value=2000.0,
        value=8.0,
        step=0.1
    )

    vitamin_b6 = st.number_input(
        "Vitamin B6",
        min_value=0.0,
        max_value=500.0,
        value=2.5,
        step=0.1
    )

    dietary_iron = st.number_input(
        "Dietary Iron",
        min_value=0.0,
        max_value=200.0,
        value=28.0,
        step=0.1
    )

    st.markdown("---")

    st.caption(
        "Input values should be obtained from available "
        "health records, laboratory reports or dietary assessment."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">🌿 FolateCare AI</div>
<div class="hero-subtitle">
AI-Based Folate Deficiency Risk Prediction in Pregnant Women
</div>
<div class="hero-description">
Machine Learning + Explainable Artificial Intelligence
</div>
<div class="hero-badge">
✨ Smart Early Risk Screening
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
<div class="info-box">
💡 <b>FolateCare AI</b> estimates the predicted risk of lower folate
status using maternal, hematological and nutritional factors.
The system also uses Explainable AI techniques to show factors
that influenced the model prediction.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM CAPABILITIES
# ============================================================

st.markdown(
    '<div class="section-title">✨ System Capabilities</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">👩</div>
<div class="feature-title">Maternal Factors</div>
<div class="feature-text">
Age and BMI are included as maternal and anthropometric
information for the prediction model.
</div>
</div>
""",
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🩸</div>
<div class="feature-title">Blood Factors</div>
<div class="feature-text">
Hemoglobin, hematocrit and RBC count are used as
hematological inputs.
</div>
</div>
""",
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🥗</div>
<div class="feature-title">Nutrition Factors</div>
<div class="feature-text">
Dietary folate, folic acid, vitamin B12, vitamin B6
and iron information are included.
</div>
</div>
""",
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🧠</div>
<div class="feature-title">Explainable AI</div>
<div class="feature-text">
SHAP and LIME provide explanations for the individual
model prediction.
</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔍 Risk Assessment</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the available patient information using the sidebar "
    "and click the button below."
)

predict_clicked = st.button(
    "🚀 Generate AI Risk Prediction",
    type="primary",
    use_container_width=True
)


# ============================================================
# BEFORE PREDICTION
# ============================================================

if not predict_clicked:

    st.markdown(
        """
<div class="info-box">
👈 <b>Ready for assessment?</b><br>
Enter the patient information in the sidebar and click
<b>Generate AI Risk Prediction</b>.
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔄 How FolateCare AI Works</div>',
        unsafe_allow_html=True
    )

    w1, w2, w3, w4, w5 = st.columns(5)

    workflow = [
        ("👩", "Patient\nInformation"),
        ("⚙️", "Data\nProcessing"),
        ("🤖", "ML Risk\nPrediction"),
        ("📊", "Risk\nProbability"),
        ("🧠", "SHAP + LIME\nExplanation")
    ]

    for col, (icon, title) in zip(
        [w1, w2, w3, w4, w5],
        workflow
    ):

        with col:

            st.markdown(
                f"""
<div class="feature-card" style="text-align:center;">
<div class="feature-icon">{icon}</div>
<div class="feature-title">{title}</div>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown(
        """
<div class="footer">
🌿 <b>FolateCare AI</b><br>
AI-Based Folate Deficiency Risk Prediction in Pregnant Women<br>
Research & Educational Prototype — Not a Medical Diagnosis
</div>
""",
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame(
    [[
        age,
        bmi,
        hgb,
        hct,
        rbc,
        dietary_folate,
        dietary_folic_acid,
        vitamin_b12,
        vitamin_b6,
        dietary_iron
    ]],
    columns=FEATURES
)


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

input_data = input_data.fillna(MEDIANS)


# ============================================================
# MODEL PREDICTION
# ============================================================

prediction = int(
    model.predict(input_data)[0]
)

probabilities = model.predict_proba(
    input_data
)[0]

risk_probability = float(
    probabilities[1]
) * 100


# ============================================================
# PREDICTION RESULT
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📊 AI Prediction Result</div>',
    unsafe_allow_html=True
)

if prediction == 1:

    st.markdown(
        f"""
<div class="high-risk">
<div class="risk-heading">🔴 HIGHER RISK</div>
<div class="risk-probability">
Estimated Risk Probability: {risk_probability:.2f}%
</div>
<div class="risk-description">
The model classified this input as having a higher predicted
risk of lower folate status.
</div>
</div>
""",
        unsafe_allow_html=True
    )

else:

    st.markdown(
        f"""
<div class="low-risk">
<div class="risk-heading">🟢 LOWER RISK</div>
<div class="risk-probability">
Estimated Risk Probability: {risk_probability:.2f}%
</div>
<div class="risk-description">
The model classified this input as having a lower predicted
risk of lower folate status.
</div>
</div>
""",
        unsafe_allow_html=True
    )


st.caption(
    "The displayed probability is the model's estimated probability. "
    "It is not a laboratory measurement or medical diagnosis."
)


# ============================================================
# PATIENT INPUT SUMMARY
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📋 Patient Input Summary</div>',
    unsafe_allow_html=True
)

summary_df = pd.DataFrame(
    {
        "Factor": [
            FEATURE_LABELS[x]
            for x in FEATURES
        ],
        "Entered Value": input_data.iloc[0].values
    }
)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

p1, p2, p3 = st.columns(3)

with p1:

    st.metric(
        "Random Forest Accuracy",
        "73.47%"
    )

with p2:

    st.metric(
        "ROC-AUC",
        "0.701"
    )

with p3:

    st.metric(
        "F1 Score",
        "51.85%"
    )

st.caption(
    "These performance values are from the project's held-out "
    "test set containing 49 samples."
)


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🧠 Explainable AI — SHAP</div>',
    unsafe_allow_html=True
)

st.write(
    "SHAP explains how each feature contributed to this individual "
    "model prediction. SHAP values describe model behavior and "
    "do not establish medical causation."
)

try:

    shap_explainer = shap.TreeExplainer(model)

    shap_values = shap_explainer.shap_values(
        input_data
    )

    shap_array = np.asarray(
        shap_values
    )

    if shap_array.ndim == 3:

        shap_row = shap_array[0, :, 1]

    elif shap_array.ndim == 2:

        shap_row = shap_array[0]

    else:

        shap_row = shap_array.flatten()

    shap_df = pd.DataFrame(
        {
            "Feature": [
                FEATURE_LABELS[x]
                for x in FEATURES
            ],
            "Value": input_data.iloc[0].values,
            "SHAP Value": shap_row
        }
    )

    shap_df["Absolute SHAP"] = (
        shap_df["SHAP Value"].abs()
    )

    shap_df = shap_df.sort_values(
        "Absolute SHAP",
        ascending=False
    )

    st.dataframe(
        shap_df[
            [
                "Feature",
                "Value",
                "SHAP Value"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    plot_df = shap_df.sort_values(
        "SHAP Value"
    )

    ax.barh(
        plot_df["Feature"],
        plot_df["SHAP Value"]
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "SHAP Value"
    )

    ax.set_title(
        "Individual Patient SHAP Explanation"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

except Exception as e:

    st.warning(
        f"SHAP explanation could not be generated: {e}"
    )


# ============================================================
# LIME EXPLANATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔎 Explainable AI — LIME</div>',
    unsafe_allow_html=True
)

st.write(
    "LIME provides a local explanation showing which features "
    "influenced this particular patient prediction."
)

try:

    X_train = train_df[
        FEATURES
    ].copy()

    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=FEATURES,
        class_names=[
            "Lower Risk",
            "Higher Risk"
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    def lime_predict(values):

        values_df = pd.DataFrame(
            values,
            columns=FEATURES
        )

        values_df = values_df.fillna(
            MEDIANS
        )

        return model.predict_proba(
            values_df
        )

    lime_exp = lime_explainer.explain_instance(
        input_data.iloc[0].values,
        lime_predict,
        num_features=10,
        labels=[prediction]
    )

    lime_list = lime_exp.as_list(
        label=prediction
    )

    lime_df = pd.DataFrame(
        lime_list,
        columns=[
            "Feature Condition",
            "Contribution"
        ]
    )

    st.dataframe(
        lime_df,
        use_container_width=True,
        hide_index=True
    )

except Exception as e:

    st.warning(
        f"LIME explanation could not be generated: {e}"
    )


# ============================================================
# GENERAL NUTRITION GUIDANCE
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🥗 General Nutrition Guidance</div>',
    unsafe_allow_html=True
)

if prediction == 1:

    st.warning(
        """
        The model indicates a higher predicted risk. This result can
        be considered an early screening signal. Further clinical
        assessment or laboratory testing may be considered with a
        qualified healthcare professional.
        """
    )

else:

    st.success(
        """
        The model indicates a lower predicted risk based on the
        entered information. This does not confirm adequate folate
        status and does not replace routine pregnancy care or
        laboratory assessment.
        """
    )

st.markdown(
    """
**General nutrient-rich food examples:**

🥬 Green leafy vegetables  
🫘 Beans and lentils  
🍊 Citrus fruits  
🌾 Fortified grain products  
🥗 Other nutrient-rich foods as part of a balanced diet  

Individual nutritional requirements during pregnancy should be
discussed with a qualified healthcare professional.
"""
)


# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📄 Assessment Report</div>',
    unsafe_allow_html=True
)

report = f"""
FOLATECARE AI
AI-Based Folate Deficiency Risk Prediction in Pregnant Women
==============================================================

PREDICTION RESULT
-----------------
Risk Level: {"HIGHER RISK" if prediction == 1 else "LOWER RISK"}
Estimated Risk Probability: {risk_probability:.2f}%

PATIENT INFORMATION
-------------------
Age: {age}
BMI: {bmi}
Hemoglobin: {hgb}
Hematocrit: {hct}
RBC Count: {rbc}

NUTRITIONAL INFORMATION
-----------------------
Dietary Folate: {dietary_folate}
Dietary Folic Acid: {dietary_folic_acid}
Vitamin B12: {vitamin_b12}
Vitamin B6: {vitamin_b6}
Dietary Iron: {dietary_iron}

MODEL INFORMATION
-----------------
Model: Random Forest
Test Accuracy: 73.47%
ROC-AUC: 0.701
F1 Score: 51.85%

IMPORTANT NOTE
--------------
This system is a research and educational prototype.
It does not diagnose folate deficiency, replace laboratory
testing, or provide medical advice.
"""

st.download_button(
    label="📥 Download Assessment Report",
    data=report,
    file_name="FolateCare_AI_Assessment_Report.txt",
    mime="text/plain",
    use_container_width=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.markdown(
    """
<div class="info-box">
⚠️ <b>Important:</b> FolateCare AI is a research and educational
prototype. The prediction should not be used as a medical diagnosis
or as a replacement for laboratory testing or professional medical
advice.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
🌿 <b>FolateCare AI</b><br>
AI-Based Folate Deficiency Risk Prediction in Pregnant Women<br>
Machine Learning + Explainable AI<br><br>
Research & Educational Prototype — Not a Medical Diagnosis
</div>
""",
    unsafe_allow_html=True
)