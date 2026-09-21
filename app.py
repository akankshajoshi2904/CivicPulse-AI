import streamlit as st
import pandas as pd
import sqlite3
import re
from datetime import datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ============================================================
# CIVICPULSE AI
# AI-powered civic problem detection and community intelligence
# ============================================================

st.set_page_config(
    page_title="CivicPulse AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "civicpulse.db"
TRAINING_PATH = BASE_DIR / "training_data.csv"
EVALUATION_PATH = BASE_DIR / "evaluation_data.csv"

CATEGORIES = [
    "Garbage / Waste",
    "Public Safety",
    "Road / Pothole",
    "Streetlight",
    "Traffic",
    "Water Supply",
    "Waterlogging / Drainage",
]

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #eef4ff, #f7f9fc);
        border: 1px solid #dfe7f3;
        margin-bottom: 1.5rem;
    }
    .hero h1 {margin-bottom: .2rem;}
    .small-muted {color: #6b7280; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Data / database helpers
# ------------------------------------------------------------
def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem TEXT NOT NULL,
            location TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            confidence REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()


def load_reports():
    connection = get_connection()
    reports = pd.read_sql_query(
        "SELECT * FROM reports ORDER BY id DESC",
        connection,
    )
    connection.close()
    return reports


initialize_database()

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
@st.cache_resource
def train_model():
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"training_data.csv was not found at {TRAINING_PATH}"
        )

    training_data = pd.read_csv(TRAINING_PATH)

    required_columns = {"problem", "category"}
    if not required_columns.issubset(training_data.columns):
        raise ValueError(
            "training_data.csv must contain 'problem' and 'category' columns."
        )

    training_data = training_data.dropna(subset=["problem", "category"]).copy()
    training_data["problem"] = training_data["problem"].astype(str)
    training_data["category"] = training_data["category"].astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(training_data["problem"])
    y = training_data["category"]

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)

    return vectorizer, model, training_data


vectorizer, model, training_data = train_model()


def predict_category(problem):
    transformed = vectorizer.transform([problem])
    probabilities = model.predict_proba(transformed)[0]
    index = probabilities.argmax()
    category = model.classes_[index]
    confidence = float(probabilities[index] * 100)
    return category, confidence


# ------------------------------------------------------------
# Risk / severity
# ------------------------------------------------------------
HIGH_RISK_WORDS = [
    "accident",
    "injury",
    "injured",
    "danger",
    "dangerous",
    "fire",
    "collision",
    "crash",
    "electrocution",
    "sewage",
    "flood",
    "flooded",
    "overflowing",
    "blocked",
    "collapse",
    "collapsed",
]

MEDIUM_RISK_WORDS = [
    "deep",
    "large",
    "major",
    "severe",
    "broken",
    "unsafe",
    "dark",
    "slow",
    "congestion",
    "waste",
    "leak",
    "crack",
    "damaged",
    "piling",
    "standing water",
]


def calculate_risk(problem, category, confidence):
    text = problem.lower()
    score = 20

    score += sum(12 for word in HIGH_RISK_WORDS if word in text)
    score += sum(5 for word in MEDIUM_RISK_WORDS if word in text)

    category_bonus = {
        "Public Safety": 12,
        "Waterlogging / Drainage": 8,
        "Road / Pothole": 6,
        "Traffic": 5,
        "Streetlight": 3,
        "Water Supply": 3,
        "Garbage / Waste": 2,
    }

    score += category_bonus.get(category, 0)

    if confidence < 45:
        score += 3

    return min(100, int(score))


def severity_from_risk(score):
    if score >= 75:
        return "Critical"
    if score >= 55:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


# ------------------------------------------------------------
# Similarity / hotspot
# ------------------------------------------------------------
def find_similar_reports(problem, threshold=0.20):
    reports = load_reports()

    if reports.empty:
        return reports

    corpus = reports["problem"].astype(str).tolist()

    similarity_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    matrix = similarity_vectorizer.fit_transform(corpus + [problem])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]

    reports["similarity"] = scores

    similar = reports[
        reports["similarity"] >= threshold
    ].sort_values("similarity", ascending=False)

    return similar.head(5)


def save_report(problem, location, category, severity, risk_score, confidence):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO reports
        (problem, location, category, severity, risk_score, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            problem,
            location,
            category,
            severity,
            int(risk_score),
            float(confidence),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    connection.commit()
    connection.close()


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
@st.cache_data
def evaluate_model():
    if not EVALUATION_PATH.exists():
        return None

    evaluation_data = pd.read_csv(EVALUATION_PATH)
    required_columns = {"problem", "category"}

    if not required_columns.issubset(evaluation_data.columns):
        return None

    evaluation_data = evaluation_data.dropna(
        subset=["problem", "category"]
    ).copy()

    X_eval = vectorizer.transform(
        evaluation_data["problem"].astype(str)
    )
    y_true = evaluation_data["category"].astype(str)
    y_pred = model.predict(X_eval)

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CATEGORIES,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CATEGORIES,
    )

    metrics_df = pd.DataFrame(
        {
            "Category": CATEGORIES,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support,
        }
    )

    return {
        "accuracy": accuracy,
        "samples": len(evaluation_data),
        "metrics": metrics_df,
        "matrix": matrix,
        "predictions": pd.DataFrame(
            {
                "problem": evaluation_data["problem"],
                "actual": y_true,
                "predicted": y_pred,
            }
        ),
    }


evaluation = evaluate_model()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.title("🏙️ CivicPulse AI")
    st.write("AI-powered early detection of emerging local problems.")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🚨 Report Problem", "📊 Community Dashboard"],
    )

    st.divider()
    st.caption("Built with Python • Streamlit • Scikit-learn • SQLite")
    st.caption("ML: TF-IDF + Logistic Regression")


# ------------------------------------------------------------
# Report Problem
# ------------------------------------------------------------
if page == "🚨 Report Problem":
    st.markdown(
        """
        <div class="hero">
            <h1>🚨 Report a Civic Problem</h1>
            <p>
                Describe a local problem. CivicPulse AI will classify it,
                estimate risk, compare it with previous reports, and look
                for repeated local patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    problem = st.text_area(
        "Describe the problem",
        placeholder=(
            "Example: There is a deep pothole near the metro station "
            "and vehicles are having difficulty passing."
        ),
        height=130,
    )

    location = st.text_input(
        "Location",
        placeholder="Example: Sector 15, Gurugram",
    )

    st.info(
        "💡 You do NOT need to select a category. "
        "CivicPulse AI will predict it automatically."
    )

    analyze = st.button(
        "🔎 Analyze & Submit Report",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        if not problem.strip():
            st.error("Please describe the civic problem.")
        elif not location.strip():
            st.error("Please enter the location.")
        else:
            category, confidence = predict_category(problem)
            risk_score = calculate_risk(
                problem,
                category,
                confidence,
            )
            severity = severity_from_risk(risk_score)

            similar_reports = find_similar_reports(problem)

            save_report(
                problem,
                location,
                category,
                severity,
                risk_score,
                confidence,
            )

            st.success("✅ Report analyzed and saved successfully!")

            st.divider()

            st.header("🧠 AI Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.caption("Detected Category")
                st.subheader(category)

            with col2:
                st.caption("Risk Score")
                st.subheader(f"{risk_score}/100")
                st.progress(risk_score / 100)

            with col3:
                st.caption("Severity")
                if severity == "Critical":
                    st.error(severity)
                elif severity == "High":
                    st.warning(severity)
                elif severity == "Medium":
                    st.info(severity)
                else:
                    st.success(severity)

            st.write(f"**AI confidence:** {confidence:.1f}%")

            st.divider()

            left, right = st.columns(2)

            with left:
                st.subheader("🔎 Similar Reports")

                if similar_reports.empty:
                    st.info("No similar previous reports found.")
                else:
                    st.write(
                        f"Found **{len(similar_reports)}** similar report(s)."
                    )

                    for _, row in similar_reports.iterrows():
                        st.write(
                            f"• {row['problem']} — **{row['location']}**"
                        )

            with right:
                st.subheader("🚨 Emerging Problem Signal")

                local_reports = similar_reports[
                    similar_reports["location"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    == location.strip().lower()
                ]

                category_reports = local_reports[
                    local_reports["category"].astype(str) == category
                ]

                category_count = len(category_reports)
                local_count = len(local_reports)

                if category_count >= 3:
                    st.error(
                        "HIGH ALERT — Repeated reports detected for "
                        "the same problem in this location."
                    )
                    st.write(
                        f"{category_count} similar {category} reports "
                        "were found at this location."
                    )
                    st.write(
                        "This pattern may indicate an emerging local "
                        "problem that needs attention."
                    )
                elif category_count == 2:
                    st.warning("WATCH — Repeated reports detected.")
                    st.write(
                        f"{category_count} similar reports for {category} "
                        "were found at this location."
                    )
                    st.write(
                        "Additional reports could confirm whether this "
                        "is becoming a recurring local problem."
                    )
                elif local_count >= 1:
                    st.info(
                        "EARLY SIGNAL — A related report exists in "
                        "this location."
                    )
                    st.write(
                        "The system found a related report at this "
                        "location, but there is not enough repeated "
                        "evidence for a hotspot yet."
                    )
                else:
                    st.success("NO EMERGING SIGNAL")
                    st.write(
                        "Similar reports were found elsewhere, but no "
                        "repeated problem was detected at this location."
                    )

            st.divider()

            st.subheader("💡 AI Recommendation")

            if risk_score >= 75:
                st.error(
                    "Immediate attention recommended. "
                    "The combination of problem type and risk indicators "
                    "suggests a high-priority issue."
                )
            elif risk_score >= 55:
                st.warning(
                    "This issue should be monitored closely and "
                    "investigated if additional reports appear."
                )
            else:
                st.info(
                    "Monitor the issue. Additional community reports "
                    "can help determine whether it is becoming widespread."
                )


# ------------------------------------------------------------
# Community Dashboard
# ------------------------------------------------------------
else:
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Community Problem Dashboard</h1>
            <p>
                Aggregated civic intelligence generated from submitted
                community reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reports = load_reports()

    if reports.empty:
        st.info("No reports have been submitted yet.")
    else:
        critical_count = int((reports["severity"] == "Critical").sum())
        high_count = int((reports["severity"] == "High").sum())
        average_risk = int(round(reports["risk_score"].mean()))

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total Reports", len(reports))

        with m2:
            st.metric("Critical Issues", critical_count)

        with m3:
            st.metric("High-Risk Issues", high_count)

        with m4:
            st.metric("Average Risk", f"{average_risk}/100")

        st.divider()

        chart1, chart2 = st.columns(2)

        with chart1:
            st.subheader("📈 Reports by Category")
            category_counts = (
                reports["category"]
                .value_counts()
                .reindex(CATEGORIES, fill_value=0)
            )
            st.bar_chart(category_counts)

        with chart2:
            st.subheader("🚨 Risk Distribution")
            risk_counts = (
                reports["severity"]
                .value_counts()
                .reindex(
                    ["Critical", "High", "Medium", "Low"],
                    fill_value=0,
                )
            )
            st.bar_chart(risk_counts)

        st.divider()

        st.subheader("📍 Location Signals")

        location_summary = (
            reports.groupby(["location", "category"])
            .size()
            .reset_index(name="reports")
            .sort_values("reports", ascending=False)
        )

        if not location_summary.empty:
            st.dataframe(
                location_summary.head(15),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        st.subheader("🔥 Highest-Risk Problems")

        highest_risk = reports.sort_values(
            ["risk_score", "created_at"],
            ascending=[False, False],
        ).head(10)

        st.dataframe(
            highest_risk[
                [
                    "problem",
                    "location",
                    "category",
                    "severity",
                    "risk_score",
                    "confidence",
                    "created_at",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # ----------------------------------------------------
        # ML evaluation
        # ----------------------------------------------------
        st.header("🤖 ML Model Evaluation")

        if evaluation is None:
            st.warning(
                "evaluation_data.csv was not found or has invalid columns."
            )
        else:
            e1, e2, e3 = st.columns(3)

            with e1:
                st.metric(
                    "Evaluation Accuracy",
                    f"{evaluation['accuracy'] * 100:.2f}%",
                )

            with e2:
                st.metric(
                    "Evaluation Samples",
                    evaluation["samples"],
                )

            with e3:
                st.metric(
                    "Training Samples",
                    len(training_data),
                )

            st.caption(
                "The classifier is evaluated on a separate evaluation "
                "dataset that was not used during model training."
            )

            metrics_display = evaluation["metrics"].copy()

            for column in ["Precision", "Recall", "F1-Score"]:
                metrics_display[column] = metrics_display[column].round(3)

            st.dataframe(
                metrics_display,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("🧩 Confusion Matrix")

            matrix_df = pd.DataFrame(
                evaluation["matrix"],
                index=CATEGORIES,
                columns=CATEGORIES,
            )

            st.dataframe(
                matrix_df,
                use_container_width=True,
            )

            with st.expander("🔬 View evaluation predictions"):
                st.dataframe(
                    evaluation["predictions"],
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        st.subheader("🧠 About the AI")

        st.write(
            """
            CivicPulse AI uses TF-IDF to convert civic problem descriptions
            into numerical text features and Logistic Regression to classify
            them into civic issue categories. Cosine similarity is used to
            find related historical reports. Risk scoring combines detected
            issue category, risk-related language, and model confidence.
            """
        )

        st.caption(
            "Current data is based on submitted reports and local project "
            "datasets. It does not represent live government or external "
            "civic data."
        )
