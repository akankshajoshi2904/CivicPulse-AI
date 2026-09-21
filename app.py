import streamlit as st
import pandas as pd
import sqlite3
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

training_data = pd.read_csv("training_data.csv")

training_text = training_data["problem"].astype(str).tolist()
training_labels = training_data["category"].astype(str).tolist()
# -------------------- TRAIN / TEST SPLIT --------------------

X_train_text, X_test_text, y_train, y_test = train_test_split(
    training_text,
    training_labels,
    test_size=0.20,
    random_state=42,
    stratify=training_labels
)


# -------------------- TF-IDF --------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# -------------------- TRAIN AI MODEL --------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)


# -------------------- MODEL EVALUATION --------------------

test_predictions = model.predict(X_test)

model_accuracy = accuracy_score(
    y_test,
    test_predictions
)

model_report = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0
)

model_confusion_matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=model.classes_
)


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


def normalize_text(value):
    return str(value).strip().lower()


def get_priority_label(severity):
    return {
        "Critical": "Urgent",
        "High": "High",
        "Medium": "Medium",
        "Low": "Normal"
    }.get(severity, severity)


def get_priority_reason(category, severity):
    category_reasons = {
        "Road / Pothole": "The report describes a road condition that may affect vehicles or pedestrians.",
        "Waterlogging / Drainage": "The report describes water or drainage problems that may affect movement and safety.",
        "Public Safety": "The report contains a public-safety concern that may need attention.",
        "Traffic": "The report describes a traffic-related problem that may affect daily travel.",
        "Streetlight": "The report describes a lighting problem that may reduce visibility in the area.",
        "Garbage / Waste": "The report describes a waste-related problem that may affect the local area.",
        "Water Supply": "The report describes a water-supply problem affecting the community."
    }

    base = category_reasons.get(
        category,
        "The report contains information about a local community problem."
    )

    if severity in ["Critical", "High"]:
        return base + " The report has been marked for higher attention."
    if severity == "Medium":
        return base + " The issue may need monitoring if more reports appear."
    return base + " More community reports can help show whether the issue is recurring."


def get_area_signals(reports):
    if reports.empty:
        return pd.DataFrame(columns=["location", "category", "reports"])

    signals = reports.copy()
    signals["location_key"] = signals["location"].astype(str).str.strip().str.lower()

    grouped = (
        signals.groupby(["location_key", "category"], as_index=False)
        .agg(
            location=("location", "first"),
            reports=("id", "count")
        )
    )

    return (
        grouped[grouped["reports"] >= 2]
        .sort_values(["reports", "location"], ascending=[False, True])
        .reset_index(drop=True)
    )


# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.title("🌆 CivicPulse AI")

    st.write(
        "Report local problems.\n\n"
        "Understand recurring issues.\n\n"
        "Help your community spot problems early."
    )

    st.divider()

    page = st.radio(
        "Go to",
        [
            "🚨 Report Problem",
            "📊 Community Dashboard",
        ]
    )


# ============================================================
# REPORT PAGE
# ============================================================

if page == "🚨 Report Problem":

    st.title("🌆 CivicPulse AI")

    st.subheader("Report a problem in your area")

    st.write(
        "Tell us what is happening and where. "
        "CivicPulse will organize the report and look for repeated problems in the area."
    )

    st.divider()

    st.header("📍 Submit a Report")

    problem = st.text_area(
        "What is happening?",
        placeholder=(
            "Example: There is a large pothole near the metro station "
            "and vehicles are having difficulty passing."
        ),
        height=150
    )

    location = st.text_input(
        "Where is it happening?",
        placeholder="Example: Sector 56, Gurugram"
    )

    st.caption("You do not need to choose a problem category. The system identifies it from your description.")

    if st.button(
        "Submit Report",
        type="primary",
        use_container_width=True
    ):

        if not problem.strip():
            st.warning("Please describe the problem.")

        elif not location.strip():
            st.warning("Please enter the location.")

        else:

            with st.spinner("Reviewing your report..."):

                category, _ = predict_category(problem)

                risk_score, severity = calculate_risk(
                    problem,
                    category
                )

                similar_reports = find_similar_reports(problem)

                save_report(
                    problem,
                    location,
                    category,
                    severity,
                    risk_score
                )

                # Load the updated database so the current report is included
                # in the repeated-problem signal.
                updated_reports = load_reports()

            location_key = normalize_text(location)
            same_area = updated_reports[
                updated_reports["location"].map(normalize_text) == location_key
            ]
            same_problem = same_area[
                same_area["category"] == category
            ]
            same_problem_count = len(same_problem)

            st.success("Your report has been added to the community records.")

            st.divider()

            st.header("What we found")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Problem type", category)

            with col2:
                st.metric("Priority", get_priority_label(severity))

            with col3:
                st.metric("Reports in this area", same_problem_count)

            st.write("**Why this priority?**")
            st.write(get_priority_reason(category, severity))

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔎 Related reports")

                if similar_reports.empty:
                    st.info(
                        "No closely related reports were found yet. "
                        "New reports may reveal a pattern over time."
                    )
                else:
                    st.write(
                        f"{len(similar_reports)} related report(s) were found in the community records."
                    )
                    st.caption(
                        "These reports are used to identify recurring problems; they are not treated as proof that the reports are the same incident."
                    )

            with col2:
                st.subheader("📍 Area signal")

                if same_problem_count >= 3:
                    st.error(
                        f"Repeated problem detected — {same_problem_count} reports for {category} in this area."
                    )
                    st.write(
                        "This is a strong recurring-report signal and may deserve local attention."
                    )

                elif same_problem_count == 2:
                    st.warning(
                        "Early recurring signal — 2 reports for this problem type were found in this area."
                    )
                    st.write(
                        "More reports can help confirm whether this is becoming a recurring local issue."
                    )

                else:
                    st.success("No repeated problem signal yet.")
                    st.write(
                        "This report is currently the only report for this problem type in this area."
                    )

            st.divider()

            st.subheader("💡 What happens next?")
            st.write(
                "Your report is now part of the community dataset. "
                "As more people report problems, CivicPulse can highlight repeated issues and areas that may need attention."
            )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Community Dashboard":

    st.title("📊 Community Problem Dashboard")

    st.write(
        "A simple view of the problems people have reported and the areas where issues are recurring."
    )

    reports = load_reports()

    if reports.empty:

        st.info(
            "No reports yet. Submit the first community report to start building local insights."
        )

    else:

        st.subheader("🔎 Explore community reports")

        filter_col1, filter_col2 = st.columns(2)

        # Clean area names for display and avoid duplicate entries caused by
        # accidental spaces/capitalization differences in submitted reports.
        area_map = {}
        for raw_area in reports["location"].dropna().astype(str):
            clean_area = raw_area.strip()
            area_map.setdefault(normalize_text(clean_area), clean_area)

        area_options = ["All areas"] + sorted(area_map.values(), key=str.lower)

        category_map = {}
        for raw_category in reports["category"].dropna().astype(str):
            clean_category = raw_category.strip()
            category_map.setdefault(normalize_text(clean_category), clean_category)

        category_options = ["All problem types"] + sorted(
            category_map.values(), key=str.lower
        )

        with filter_col1:
            selected_area = st.selectbox(
                "Area",
                area_options
            )

        with filter_col2:
            selected_category = st.selectbox(
                "Problem type",
                category_options
            )

        filtered_reports = reports.copy()

        if selected_area != "All areas":
            selected_area_key = normalize_text(selected_area)
            filtered_reports = filtered_reports[
                filtered_reports["location"].map(normalize_text) == selected_area_key
            ]

        if selected_category != "All problem types":
            filtered_reports = filtered_reports[
                filtered_reports["category"] == selected_category
            ]

        area_signals = get_area_signals(filtered_reports)

        total_reports = len(filtered_reports)
        repeated_problem_groups = len(area_signals)
        areas_to_watch = area_signals["location"].nunique() if not area_signals.empty else 0
        higher_priority = len(
            filtered_reports[filtered_reports["severity"].isin(["High", "Critical"])]
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Reports received", total_reports)

        with col2:
            st.metric("Repeated problems", repeated_problem_groups)

        with col3:
            st.metric("Areas to watch", areas_to_watch)

        with col4:
            st.metric("Higher-priority reports", higher_priority)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 What people are reporting")

            category_counts = (
                filtered_reports["category"]
                .value_counts()
                .rename_axis("Problem")
                .to_frame("Reports")
            )

            st.bar_chart(category_counts)

        with col2:
            st.subheader("🚨 Reports by priority")

            priority_counts = (
                filtered_reports["severity"]
                .map(get_priority_label)
                .value_counts()
                .reindex(["Normal", "Medium", "High", "Urgent"], fill_value=0)
                .rename_axis("Priority")
                .to_frame("Reports")
            )

            st.bar_chart(priority_counts)

        st.divider()

        st.subheader("📍 Areas with repeated problems")

        if area_signals.empty:
            st.info(
                "No area has received multiple reports for the same problem type yet."
            )
        else:
            display_signals = area_signals[
                ["location", "category", "reports"]
            ].copy()
            display_signals.columns = [
                "Area",
                "Problem",
                "Reports"
            ]
            st.dataframe(
                display_signals.head(10),
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.subheader("⚠️ Problems that may need attention")

        attention = filtered_reports[
            filtered_reports["severity"].isin(["High", "Critical"])
        ].sort_values(
            ["risk_score", "created_at"],
            ascending=[False, False]
        ).head(10).copy()

        if attention.empty:
            st.success(
                "No reports are currently marked High or Urgent."
            )
        else:
            attention["Priority"] = attention["severity"].map(get_priority_label)
            attention = attention[
                ["problem", "location", "category", "Priority", "created_at"]
            ]
            attention.columns = [
                "Reported problem",
                "Area",
                "Problem type",
                "Priority",
                "Reported at"
            ]
            st.dataframe(
                attention,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.subheader("🕒 Recent community reports")

        recent = filtered_reports.head(10).copy()
        recent = recent[
            ["problem", "location", "category", "severity", "created_at"]
        ]
        recent["severity"] = recent["severity"].map(get_priority_label)
        recent.columns = [
            "Reported problem",
            "Area",
            "Problem type",
            "Priority",
            "Reported at"
        ]

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("💬 What CivicPulse does")
        st.write(
            "CivicPulse AI turns individual community reports into useful local insights. "
            "It organizes reported problems, looks for repeated issues, and highlights areas where multiple reports may indicate a recurring concern."
        )

        st.caption(
            "Community signals are based on reports submitted to this application and should be treated as indicators, not official government assessments."
        )
