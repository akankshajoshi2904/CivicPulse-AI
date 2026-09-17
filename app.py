import streamlit as st
import pandas as pd
import sqlite3
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CIVICPULSE AI
# AI-powered early detection of local problems
# ============================================================


# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="CivicPulse AI",
    page_icon="🌆",
    layout="wide"
)


# -------------------- DATABASE --------------------

def create_database():

    connection = sqlite3.connect("civicpulse.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem TEXT,
            location TEXT,
            category TEXT,
            severity TEXT,
            risk_score INTEGER,
            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()


create_database()


# -------------------- TRAINING DATA --------------------

training_text = [

    # Road / pothole
    "large pothole on road",
    "road has a deep pothole",
    "broken road near school",
    "damaged road needs repair",
    "many potholes on street",
    "road surface is broken",
    "huge pothole causing accidents",
    "street road is damaged",

    # Waterlogging / drainage
    "water is collecting after rain",
    "street is flooded",
    "waterlogging near metro",
    "drain is overflowing",
    "water is not draining",
    "road is full of rain water",
    "sewage water on road",
    "drainage problem in area",

    # Streetlight
    "street light is not working",
    "road is dark at night",
    "broken street lamp",
    "streetlight has stopped working",
    "no light on the road",
    "lamp post is damaged",
    "street lights are off",

    # Garbage / waste
    "garbage is not being collected",
    "trash is lying on road",
    "garbage dump near house",
    "waste is piling up",
    "dustbins are overflowing",
    "dirty garbage area",
    "trash has not been removed",

    # Traffic
    "heavy traffic near junction",
    "traffic jam every morning",
    "cars are blocking the road",
    "intersection has severe congestion",
    "traffic signal is causing jams",
    "too many vehicles on road",

    # Water supply
    "no water supply",
    "water shortage in apartment",
    "tap has no water",
    "water supply has stopped",
    "dirty water coming from tap",
    "water pressure is very low",

    # Public safety
    "unsafe area at night",
    "someone is suspicious near the street",
    "dangerous area",
    "people feel unsafe here",
    "crime risk is increasing",
    "broken railing is dangerous"
]

training_labels = [

    "Road / Pothole",
    "Road / Pothole",
    "Road / Pothole",
    "Road / Pothole",
    "Road / Pothole",
    "Road / Pothole",
    "Road / Pothole",
    "Road / Pothole",

    "Waterlogging / Drainage",
    "Waterlogging / Drainage",
    "Waterlogging / Drainage",
    "Waterlogging / Drainage",
    "Waterlogging / Drainage",
    "Waterlogging / Drainage",
    "Waterlogging / Drainage",
    "Waterlogging / Drainage",

    "Streetlight",
    "Streetlight",
    "Streetlight",
    "Streetlight",
    "Streetlight",
    "Streetlight",
    "Streetlight",

    "Garbage / Waste",
    "Garbage / Waste",
    "Garbage / Waste",
    "Garbage / Waste",
    "Garbage / Waste",
    "Garbage / Waste",
    "Garbage / Waste",

    "Traffic",
    "Traffic",
    "Traffic",
    "Traffic",
    "Traffic",
    "Traffic",

    "Water Supply",
    "Water Supply",
    "Water Supply",
    "Water Supply",
    "Water Supply",
    "Water Supply",

    "Public Safety",
    "Public Safety",
    "Public Safety",
    "Public Safety",
    "Public Safety",
    "Public Safety"
]


# -------------------- TRAIN AI MODEL --------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(training_text)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, training_labels)


# -------------------- AI CATEGORY PREDICTION --------------------

def predict_category(problem):

    transformed_problem = vectorizer.transform([problem])

    prediction = model.predict(transformed_problem)[0]

    probabilities = model.predict_proba(transformed_problem)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# -------------------- RISK CALCULATION --------------------

def calculate_risk(problem, category):

    text = problem.lower()

    risk = 30

    high_risk_words = [
        "accident",
        "danger",
        "dangerous",
        "injury",
        "flood",
        "flooded",
        "sewage",
        "fire",
        "crime",
        "unsafe",
        "emergency",
        "blocked",
        "overflowing"
    ]

    medium_risk_words = [
        "broken",
        "deep",
        "large",
        "huge",
        "severe",
        "heavy",
        "major",
        "bad"
    ]

    for word in high_risk_words:
        if word in text:
            risk += 12

    for word in medium_risk_words:
        if word in text:
            risk += 6

    if category in ["Public Safety", "Waterlogging / Drainage"]:
        risk += 8

    risk = min(risk, 100)

    if risk >= 75:
        severity = "Critical"
    elif risk >= 55:
        severity = "High"
    elif risk >= 40:
        severity = "Medium"
    else:
        severity = "Low"

    return risk, severity


# -------------------- SIMILARITY SEARCH --------------------

def find_similar_reports(problem):

    connection = sqlite3.connect("civicpulse.db")

    reports = pd.read_sql_query(
        "SELECT * FROM reports ORDER BY id DESC",
        connection
    )

    connection.close()

    if reports.empty:
        return reports

    all_text = reports["problem"].tolist()

    similarity_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    matrix = similarity_vectorizer.fit_transform(
        all_text + [problem]
    )

    similarity_scores = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )[0]

    reports["similarity"] = similarity_scores

    similar = reports[
        reports["similarity"] >= 0.20
    ].sort_values(
        "similarity",
        ascending=False
    )

    return similar.head(5)


# -------------------- SAVE REPORT --------------------

def save_report(problem, location, category, severity, risk_score):

    connection = sqlite3.connect("civicpulse.db")

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO reports
        (problem, location, category, severity, risk_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            problem,
            location,
            category,
            severity,
            risk_score,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    connection.commit()
    connection.close()


# -------------------- LOAD REPORTS --------------------

def load_reports():

    connection = sqlite3.connect("civicpulse.db")

    reports = pd.read_sql_query(
        "SELECT * FROM reports ORDER BY id DESC",
        connection
    )

    connection.close()

    return reports


# ============================================================
# USER INTERFACE
# ============================================================


# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.title("🌆 CivicPulse AI")

    st.write(
        "AI-powered early detection of emerging "
        "local problems."
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🚨 Report Problem",
            "📊 Community Dashboard",
            "🧠 How the AI Works"
        ]
    )

    st.divider()

    st.caption(
        "Built with Python • Streamlit • "
        "Scikit-learn • SQLite"
    )


# ============================================================
# REPORT PAGE
# ============================================================

if page == "🚨 Report Problem":

    st.title("🌆 CivicPulse AI")

    st.subheader(
        "AI-Powered Early Detection of Local Problems"
    )

    st.write(
        "Report a problem in your area. "
        "Our AI analyzes the report and checks "
        "whether it may be part of an emerging local hotspot."
    )

    st.divider()

    st.header("📍 Report a Problem")

    problem = st.text_area(
        "What is happening?",
        placeholder=(
            "Example: Water has been collecting outside "
            "the metro station after rain and pedestrians "
            "are struggling to cross the road."
        ),
        height=150
    )

    location = st.text_input(
        "Where is it happening?",
        placeholder="Example: Sector 56, Gurugram"
    )

    st.info(
        "💡 You do NOT need to select a category. "
        "CivicPulse AI will predict it automatically."
    )

    if st.button(
        "🤖 Analyze & Submit Report",
        type="primary",
        use_container_width=True
    ):

        if problem.strip() == "":
            st.warning("Please describe the problem.")

        elif location.strip() == "":
            st.warning("Please enter the location.")

        else:

            with st.spinner("🤖 AI is analyzing the report..."):

                category, confidence = predict_category(problem)

                risk_score, severity = calculate_risk(
                    problem,
                    category
                )

                similar_reports = find_similar_reports(
                    problem
                )

                save_report(
                    problem,
                    location,
                    category,
                    severity,
                    risk_score
                )

            st.success(
                "✅ Report analyzed and saved successfully!"
            )

            st.divider()

            st.header("🧠 AI Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Detected Category",
                    category
                )

            with col2:
                st.metric(
                    "Risk Score",
                    f"{risk_score}/100"
                )

            with col3:
                st.metric(
                    "Severity",
                    severity
                )

            st.progress(
                risk_score / 100
            )

            st.write(
                f"**AI confidence:** {confidence:.1f}%"
            )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("🔎 Similar Reports")

                if similar_reports.empty:

                    st.write(
                        "No similar reports found yet."
                    )

                else:

                    st.write(
                        f"Found **{len(similar_reports)} "
                        "similar report(s)**."
                    )

                    for _, row in similar_reports.iterrows():

                        st.write(
                            f"• {row['problem']} "
                            f"— {row['location']}"
                        )

            with col2:

                st.subheader("🚨 Emerging Hotspot")

                if len(similar_reports) >= 3:

                    st.error(
                        "HIGH ALERT — Multiple similar "
                        "reports detected."
                    )

                    st.write(
                        "This location may represent an "
                        "emerging local problem."
                    )

                elif len(similar_reports) >= 1:

                    st.warning(
                        "Potential emerging problem."
                    )

                    st.write(
                        "More reports should be monitored."
                    )

                else:

                    st.success(
                        "No hotspot detected yet."
                    )

                    st.write(
                        "This appears to be an isolated report."
                    )

            st.divider()

            st.subheader("💡 AI Recommendation")

            if risk_score >= 75:

                st.error(
                    "Immediate attention recommended. "
                    "The combination of problem type and "
                    "risk indicators suggests a high-priority issue."
                )

            elif risk_score >= 55:

                st.warning(
                    "This issue should be monitored closely "
                    "and investigated if additional reports appear."
                )

            else:

                st.info(
                    "Monitor the issue. Additional community "
                    "reports can help determine whether it is becoming widespread."
                )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Community Dashboard":

    st.title("📊 Community Problem Dashboard")

    reports = load_reports()

    if reports.empty:

        st.info(
            "No reports yet. Submit your first report "
            "to populate the dashboard."
        )

    else:

        total_reports = len(reports)

        critical_reports = len(
            reports[reports["severity"] == "Critical"]
        )

        high_reports = len(
            reports[reports["severity"] == "High"]
        )

        average_risk = int(
            reports["risk_score"].mean()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Reports",
                total_reports
            )

        with col2:
            st.metric(
                "Critical Issues",
                critical_reports
            )

        with col3:
            st.metric(
                "High-Risk Issues",
                high_reports
            )

        with col4:
            st.metric(
                "Average Risk",
                f"{average_risk}/100"
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📈 Reports by Category")

            category_counts = (
                reports["category"]
                .value_counts()
                .reset_index()
            )

            category_counts.columns = [
                "Category",
                "Reports"
            ]

            st.bar_chart(
                category_counts.set_index("Category")
            )

        with col2:

            st.subheader("🚨 Risk Distribution")

            risk_counts = (
                reports["severity"]
                .value_counts()
                .reindex(
                    ["Low", "Medium", "High", "Critical"],
                    fill_value=0
                )
            )

            st.bar_chart(risk_counts)

        st.divider()

        st.subheader("🔥 Highest-Risk Problems")

        top_reports = reports.sort_values(
            "risk_score",
            ascending=False
        ).head(10)

        st.dataframe(
            top_reports[
                [
                    "problem",
                    "location",
                    "category",
                    "severity",
                    "risk_score",
                    "created_at"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# AI EXPLANATION
# ============================================================

elif page == "🧠 How the AI Works":

    st.title("🧠 How CivicPulse AI Works")

    st.write(
        "CivicPulse uses machine learning and text similarity "
        "to turn individual citizen observations into useful "
        "community-level signals."
    )

    st.divider()

    st.header("1️⃣ AI Category Detection")

    st.write(
        "The system converts the user's description into "
        "numerical text features using TF-IDF."
    )

    st.code(
        "Problem description\n"
        "       ↓\n"
        "TF-IDF text representation\n"
        "       ↓\n"
        "Logistic Regression\n"
        "       ↓\n"
        "Predicted category"
    )

    st.divider()

    st.header("2️⃣ Risk Scoring")

    st.write(
        "The system looks for risk indicators such as "
        "danger, accident, flooding, sewage, unsafe and "
        "other signals."
    )

    st.code(
        "Risk Score = Base Risk + Risk Indicators + Category Risk"
    )

    st.divider()

    st.header("3️⃣ Similar Problem Detection")

    st.write(
        "TF-IDF vectors and cosine similarity are used to "
        "compare a new report with previous reports."
    )

    st.code(
        "New Report\n"
        "     ↓\n"
        "TF-IDF Vector\n"
        "     ↓\n"
        "Cosine Similarity\n"
        "     ↓\n"
        "Similar Reports"
    )

    st.divider()

    st.header("4️⃣ Emerging Hotspot Detection")

    st.write(
        "When several similar reports appear, CivicPulse "
        "flags the situation as a potential emerging hotspot."
    )

    st.success(
        "The goal is not simply to collect complaints. "
        "The goal is to detect patterns."
    )