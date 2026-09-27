import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from datetime import date


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
# FILE PATHS
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
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "risk_probability" not in st.session_state:
    st.session_state.risk_probability = 0.0


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -----------------------------
       MAIN BACKGROUND
    ----------------------------- */

    .stApp {
        background: #f6faf8;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    /* -----------------------------
       SIDEBAR
    ----------------------------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #e8f7ef 0%,
            #eef5ff 100%
        );
        border-right: 1px solid #d4e5dc;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #14543e;
    }


    /* -----------------------------
       BUTTONS
    ----------------------------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #c9dfd5;
    }

    .stButton > button:hover {
        border-color: #16865f;
    }


    /* -----------------------------
       HERO
    ----------------------------- */

    .hero-title {
        font-size: 42px;
        font-weight: 850;
        color: #124d3b;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 19px;
        color: #4e7065;
        line-height: 1.5;
    }

    .hero-small {
        color: #16865f;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1px;
    }


    /* -----------------------------
       CARDS
    ----------------------------- */

    .card-title {
        color: #173f66;
        font-size: 19px;
        font-weight: 800;
    }

    .card-text {
        color: #647780;
        font-size: 14px;
        line-height: 1.6;
    }


    /* -----------------------------
       RISK CARDS
    ----------------------------- */

    .high-risk {
        background: #fff1f3;
        border: 2px solid #efb5bd;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
    }

    .low-risk {
        background: #ecfbf2;
        border: 2px solid #a9ddbd;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
    }

    .risk-title {
        font-size: 32px;
        font-weight: 850;
        color: #263238;
    }

    .risk-number {
        font-size: 23px;
        font-weight: 750;
        color: #40535a;
        margin-top: 8px;
    }


    /* -----------------------------
       METRICS
    ----------------------------- */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce9e3;
        border-radius: 14px;
        padding: 15px;
        box-shadow: 0 4px 14px rgba(40, 70, 60, 0.05);
    }


    /* -----------------------------
       TABLE
    ----------------------------- */

    .stDataFrame {
        border-radius: 12px;
    }


    /* -----------------------------
       FOOTER
    ----------------------------- */

    .footer {
        text-align: center;
        color: #718096;
        font-size: 13px;
        padding-top: 30px;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🌿 FolateCare AI")
    st.caption("Pregnancy Folate Risk Screening")

    st.markdown("---")

    st.markdown("### 🧭 Navigation")

    if st.button("🏠  Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
        st.rerun()

    if st.button("👩  Patient Profile", use_container_width=True):
        st.session_state.page = "Patient Profile"
        st.rerun()

    if st.button("🩸  Health & Lab Data", use_container_width=True):
        st.session_state.page = "Health & Lab Data"
        st.rerun()

    if st.button("🥗  Nutrition", use_container_width=True):
        st.session_state.page = "Nutrition"
        st.rerun()

    if st.button("🤖  AI Prediction", use_container_width=True):
        st.session_state.page = "AI Prediction"
        st.rerun()

    if st.button("🧠  Explainable AI", use_container_width=True):
        st.session_state.page = "Explainable AI"
        st.rerun()

    if st.button("📄  Report", use_container_width=True):
        st.session_state.page = "Report"
        st.rerun()

    st.markdown("---")

    st.info(
        """
        **Research Prototype**

        This system provides AI-based
        early risk screening support.

        It is not a medical diagnosis.
        """
    )


# ============================================================
# COMMON INPUT VALUES
# ============================================================

with st.sidebar:

    st.markdown("### 👩 Patient Data")

    age = st.number_input(
        "Age",
        18.0,
        50.0,
        27.0,
        1.0
    )

    bmi = st.number_input(
        "BMI",
        10.0,
        80.0,
        24.5,
        0.1
    )

    st.markdown("### 🩸 Blood Data")

    hgb = st.number_input(
        "Hemoglobin",
        5.0,
        20.0,
        12.4,
        0.1
    )

    hct = st.number_input(
        "Hematocrit",
        15.0,
        55.0,
        36.5,
        0.1
    )

    rbc = st.number_input(
        "RBC Count",
        2.0,
        7.0,
        4.10,
        0.01
    )

    st.markdown("### 🥗 Nutrition")

    dietary_folate = st.number_input(
        "Dietary Folate",
        0.0,
        10000.0,
        1200.0,
        10.0
    )

    dietary_folic_acid = st.number_input(
        "Dietary Folic Acid",
        0.0,
        10000.0,
        700.0,
        10.0
    )

    vitamin_b12 = st.number_input(
        "Vitamin B12",
        0.0,
        2000.0,
        8.0,
        0.1
    )

    vitamin_b6 = st.number_input(
        "Vitamin B6",
        0.0,
        500.0,
        2.5,
        0.1
    )

    dietary_iron = st.number_input(
        "Dietary Iron",
        0.0,
        200.0,
        28.0,
        0.1
    )


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

input_data = input_data.fillna(MEDIANS)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def generate_prediction():

    pred = int(
        model.predict(input_data)[0]
    )

    probs = model.predict_proba(
        input_data
    )[0]

    probability = float(
        probs[1]
    ) * 100

    st.session_state.prediction = pred
    st.session_state.risk_probability = probability
    st.session_state.prediction_done = True


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        "### 🌿 FOLATECARE AI"
    )

    st.markdown(
        '<div class="hero-title">Smart Folate Risk Screening</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-subtitle">
        AI-Based Folate Deficiency Risk Prediction in Pregnant Women
        <br>
        <b>Machine Learning + Explainable Artificial Intelligence</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    st.success(
        "🌱 Early risk screening support using maternal, "
        "hematological and nutritional information."
    )

    st.markdown("---")

    st.markdown("## 📊 FolateCare AI Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            st.markdown("### 👩")
            st.markdown("**Maternal Factors**")
            st.caption(
                "Age and BMI information"
            )

    with c2:
        with st.container(border=True):
            st.markdown("### 🩸")
            st.markdown("**Blood Factors**")
            st.caption(
                "Hemoglobin, hematocrit and RBC"
            )

    with c3:
        with st.container(border=True):
            st.markdown("### 🥗")
            st.markdown("**Nutrition Factors**")
            st.caption(
                "Folate, folic acid, B12, B6 and iron"
            )

    with c4:
        with st.container(border=True):
            st.markdown("### 🧠")
            st.markdown("**Explainable AI**")
            st.caption(
                "SHAP and LIME explanations"
            )

    st.markdown("---")

    st.markdown("## 🔄 Assessment Workflow")

    w1, w2, w3, w4, w5 = st.columns(5)

    with w1:
        st.info("**01**\n\n👩 Patient")

    with w2:
        st.info("**02**\n\n🩸 Health Data")

    with w3:
        st.info("**03**\n\n🥗 Nutrition")

    with w4:
        st.info("**04**\n\n🤖 Prediction")

    with w5:
        st.info("**05**\n\n🧠 XAI")

    st.markdown("---")

    st.markdown("## 📈 Research Model Performance")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Accuracy",
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
        "Random Forest model evaluated on the project's "
        "held-out test set."
    )


# ============================================================
# PATIENT PROFILE
# ============================================================

elif st.session_state.page == "Patient Profile":

    st.markdown("## 👩 Patient Profile")

    st.caption(
        "Enter basic maternal information for the assessment."
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("### 👩 Maternal Information")

            st.metric(
                "Age",
                f"{age:.0f} years"
            )

            st.metric(
                "BMI",
                f"{bmi:.1f} kg/m²"
            )

    with c2:

        with st.container(border=True):

            st.markdown("### 🤰 Assessment Information")

            st.write(
                "**Target Population**"
            )

            st.success(
                "Pregnant Women"
            )

            st.write(
                "**Assessment Date**"
            )

            st.info(
                str(date.today())
            )

    st.markdown("---")

    st.info(
        """
        The system focuses specifically on pregnant women.
        The displayed patient information is used to support
        the AI-based folate risk assessment.
        """
    )

    if st.button(
        "Continue to Health Data →",
        type="primary",
        use_container_width=True
    ):
        st.session_state.page = "Health & Lab Data"
        st.rerun()


# ============================================================
# HEALTH AND LAB
# ============================================================

elif st.session_state.page == "Health & Lab Data":

    st.markdown("## 🩸 Health & Laboratory Data")

    st.caption(
        "Enter values available from the patient's laboratory report."
    )

    st.markdown("---")

    a, b, c = st.columns(3)

    with a:

        with st.container(border=True):

            st.markdown("### 🩸 Hemoglobin")

            st.metric(
                "Current Value",
                f"{hgb:.1f} g/dL"
            )

            st.caption(
                "Obtained from available blood report."
            )

    with b:

        with st.container(border=True):

            st.markdown("### 🧪 Hematocrit")

            st.metric(
                "Current Value",
                f"{hct:.1f} %"
            )

            st.caption(
                "Obtained from available blood report."
            )

    with c:

        with st.container(border=True):

            st.markdown("### 🔴 RBC Count")

            st.metric(
                "Current Value",
                f"{rbc:.2f}"
            )

            st.caption(
                "Obtained from available blood report."
            )

    st.markdown("---")

    st.markdown("### 📋 Laboratory Summary")

    lab_df = pd.DataFrame(
        {
            "Parameter": [
                "Hemoglobin",
                "Hematocrit",
                "RBC Count"
            ],
            "Value": [
                f"{hgb:.1f} g/dL",
                f"{hct:.1f} %",
                f"{rbc:.2f}"
            ],
            "Source": [
                "Laboratory Report",
                "Laboratory Report",
                "Laboratory Report"
            ]
        }
    )

    st.dataframe(
        lab_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Continue to Nutrition →",
        type="primary",
        use_container_width=True
    ):
        st.session_state.page = "Nutrition"
        st.rerun()


# ============================================================
# NUTRITION
# ============================================================

elif st.session_state.page == "Nutrition":

    st.markdown("## 🥗 Nutrition Assessment")

    st.caption(
        "Enter dietary assessment values available for the patient."
    )

    st.markdown("---")

    n1, n2, n3, n4, n5 = st.columns(5)

    with n1:

        with st.container(border=True):

            st.markdown("### 🌿 Folate")

            st.metric(
                "Value",
                f"{dietary_folate:.0f}"
            )

    with n2:

        with st.container(border=True):

            st.markdown("### 💊 Folic Acid")

            st.metric(
                "Value",
                f"{dietary_folic_acid:.0f}"
            )

    with n3:

        with st.container(border=True):

            st.markdown("### 🧬 Vitamin B12")

            st.metric(
                "Value",
                f"{vitamin_b12:.1f}"
            )

    with n4:

        with st.container(border=True):

            st.markdown("### 🧬 Vitamin B6")

            st.metric(
                "Value",
                f"{vitamin_b6:.1f}"
            )

    with n5:

        with st.container(border=True):

            st.markdown("### 🥩 Iron")

            st.metric(
                "Value",
                f"{dietary_iron:.1f}"
            )

    st.markdown("---")

    st.markdown("### 🥗 Nutritional Summary")

    nutrition_df = pd.DataFrame(
        {
            "Nutrient": [
                "Dietary Folate",
                "Dietary Folic Acid",
                "Vitamin B12",
                "Vitamin B6",
                "Dietary Iron"
            ],
            "Entered Value": [
                dietary_folate,
                dietary_folic_acid,
                vitamin_b12,
                vitamin_b6,
                dietary_iron
            ]
        }
    )

    st.dataframe(
        nutrition_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.info(
        """
        Nutritional information can be obtained through dietary
        assessment or available nutritional records.
        """
    )

    if st.button(
        "🤖 Generate AI Risk Prediction",
        type="primary",
        use_container_width=True
    ):

        generate_prediction()

        st.session_state.page = "AI Prediction"

        st.rerun()


# ============================================================
# AI PREDICTION
# ============================================================

elif st.session_state.page == "AI Prediction":

    st.markdown("## 🤖 AI Risk Prediction")

    st.caption(
        "Random Forest model prediction based on the entered information."
    )

    st.markdown("---")

    if not st.session_state.prediction_done:

        st.warning(
            "No prediction has been generated yet."
        )

        if st.button(
            "🚀 Generate AI Prediction",
            type="primary",
            use_container_width=True
        ):

            generate_prediction()
            st.rerun()

    else:

        prediction = st.session_state.prediction

        probability = st.session_state.risk_probability

        if prediction == 1:

            st.markdown(
                f"""
                <div class="high-risk">

                <div class="risk-title">
                🔴 HIGHER RISK
                </div>

                <div class="risk-number">
                Estimated Risk Probability: {probability:.2f}%
                </div>

                <p>
                The model classified this input as having a higher
                predicted risk of lower folate status.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="low-risk">

                <div class="risk-title">
                🟢 LOWER RISK
                </div>

                <div class="risk-number">
                Estimated Risk Probability: {probability:.2f}%
                </div>

                <p>
                The model classified this input as having a lower
                predicted risk of lower folate status.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown("### 📋 Patient Summary")

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

        st.markdown("---")

        st.markdown("### ⚠️ Important")

        st.info(
            """
            The displayed probability is the model's estimated
            probability. It is not a laboratory measurement and
            does not constitute a medical diagnosis.
            """
        )

        if st.button(
            "🧠 View Explainable AI",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = "Explainable AI"
            st.rerun()


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif st.session_state.page == "Explainable AI":

    st.markdown("## 🧠 Explainable AI")

    st.caption(
        "Understand which features influenced the individual prediction."
    )

    st.markdown("---")

    if not st.session_state.prediction_done:

        st.warning(
            "Generate a prediction first."
        )

    else:

        # ====================================================
        # SHAP
        # ====================================================

        st.markdown("### 🧠 SHAP — Feature Contribution")

        st.write(
            """
            SHAP explains how individual features contributed to
            the model prediction. These values describe model
            behavior and do not establish medical causation.
            """
        )

        try:

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(
                input_data
            )

            shap_array = np.asarray(
                shap_values
            )

            if shap_array.ndim == 3:

                shap_row = shap_array[0, :, 1]

            elif shap_array.ndim == 2:

                if shap_array.shape[0] == 2:

                    shap_row = shap_array[1]

                else:

                    shap_row = shap_array[0]

            else:

                shap_row = shap_array.flatten()

            shap_row = np.asarray(
                shap_row
            ).flatten()

            if len(shap_row) == len(FEATURES):

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

        st.markdown("---")

        # ====================================================
        # LIME
        # ====================================================

        st.markdown("### 🔎 LIME — Individual Explanation")

        st.write(
            """
            LIME provides a local explanation showing which
            features influenced this particular patient prediction.
            """
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
                labels=[
                    int(st.session_state.prediction)
                ]
            )

            lime_list = lime_exp.as_list(
                label=int(st.session_state.prediction)
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

        st.markdown("---")

        st.success(
            """
            SHAP = feature contribution for the model prediction.

            LIME = local explanation for the individual patient.
            """
        )


# ============================================================
# REPORT
# ============================================================

elif st.session_state.page == "Report":

    st.markdown("## 📄 Assessment Report")

    st.caption(
        "Summary of the current FolateCare AI assessment."
    )

    st.markdown("---")

    if not st.session_state.prediction_done:

        st.warning(
            "Generate an AI prediction before creating the report."
        )

    else:

        prediction = st.session_state.prediction

        probability = st.session_state.risk_probability

        risk_text = (
            "HIGHER RISK"
            if prediction == 1
            else
            "LOWER RISK"
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Risk Status",
                risk_text
            )

        with r2:
            st.metric(
                "Risk Probability",
                f"{probability:.2f}%"
            )

        with r3:
            st.metric(
                "Model",
                "Random Forest"
            )

        st.markdown("---")

        st.markdown("### 👩 Maternal Information")

        maternal_df = pd.DataFrame(
            {
                "Parameter": [
                    "Age",
                    "BMI"
                ],
                "Value": [
                    age,
                    bmi
                ]
            }
        )

        st.dataframe(
            maternal_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 🩸 Health Information")

        health_df = pd.DataFrame(
            {
                "Parameter": [
                    "Hemoglobin",
                    "Hematocrit",
                    "RBC Count"
                ],
                "Value": [
                    hgb,
                    hct,
                    rbc
                ]
            }
        )

        st.dataframe(
            health_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 🥗 Nutrition Information")

        food_df = pd.DataFrame(
            {
                "Nutrient": [
                    "Dietary Folate",
                    "Dietary Folic Acid",
                    "Vitamin B12",
                    "Vitamin B6",
                    "Dietary Iron"
                ],
                "Value": [
                    dietary_folate,
                    dietary_folic_acid,
                    vitamin_b12,
                    vitamin_b6,
                    dietary_iron
                ]
            }
        )

        st.dataframe(
            food_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.markdown("### 🥗 General Nutrition Guidance")

        st.info(
            """
            General nutrient-rich food examples include:

            • Green leafy vegetables

            • Beans and lentils

            • Citrus fruits

            • Fortified grain products

            • Other nutrient-rich foods as part of a balanced diet

            Individual nutritional requirements during pregnancy
            should be discussed with a qualified healthcare professional.
            """
        )

        report = f"""
FOLATECARE AI
AI-Based Folate Deficiency Risk Prediction in Pregnant Women
==============================================================

ASSESSMENT DATE
---------------
{date.today()}

PREDICTION RESULT
-----------------
Risk Level: {risk_text}
Estimated Risk Probability: {probability:.2f}%

MATERNAL INFORMATION
--------------------
Age: {age}
BMI: {bmi}

HEALTH INFORMATION
------------------
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
            "📥 Download Assessment Report",
            data=report,
            file_name="FolateCare_AI_Assessment_Report.txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    🌿 <b>FolateCare AI</b>

    <br>

    AI-Based Folate Deficiency Risk Prediction in Pregnant Women

    <br>

    Machine Learning + Explainable AI

    <br><br>

    Research & Educational Prototype — Not a Medical Diagnosis

    </div>
    """,
    unsafe_allow_html=True
)