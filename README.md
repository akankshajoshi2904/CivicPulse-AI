# CivicPulse AI

AI-powered early detection of emerging local problems from citizen reports.

CivicPulse AI allows users to report problems in their area and uses machine learning and text similarity techniques to analyze reports, identify problem categories, calculate risk, find similar reports, and detect possible emerging local hotspots.

## Features

- Submit local problem reports
- AI-based problem category detection
- Risk score calculation
- Similar problem detection
- Emerging hotspot detection
- Community problem dashboard
- SQLite-based report storage
- Simple and interactive Streamlit interface

## How It Works

A user submits a description of a local problem along with its location.

CivicPulse AI processes the report through multiple stages:

1. The problem description is converted into numerical text features using TF-IDF.
2. A Logistic Regression model predicts the problem category.
3. Risk indicators are analyzed to calculate a risk score.
4. TF-IDF vectors and cosine similarity are used to find similar previous reports.
5. Multiple similar reports can indicate a potential emerging local hotspot.
6. The report and analysis are stored in a SQLite database.
7. The community dashboard displays aggregated problem information.

## AI / Machine Learning

### Category Detection

CivicPulse AI uses:

- TF-IDF for text feature extraction
- Logistic Regression for category prediction

The model analyzes the text description and predicts a category such as:

- Road / Pothole
- Waterlogging / Drainage
- Other supported problem categories

### Similar Problem Detection

New reports are compared with previous reports using:

- TF-IDF vectorization
- Cosine similarity

This helps identify reports describing similar problems in the same or nearby areas.

### Risk Scoring

The application calculates a risk score using factors such as:

- Risk indicators in the report
- Problem category
- Other predefined risk factors

The resulting score is used to assign a severity level.

### Emerging Hotspot Detection

When several similar reports are detected, CivicPulse AI can flag the situation as a potential emerging problem.

The purpose is to identify patterns across individual reports rather than simply collecting complaints.

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- SQLite
- Pandas
- TF-IDF
- Logistic Regression
- Cosine Similarity

## Project Structure

```text
CivicPulse-AI/
│
├── app.py
├── civicpulse.db
├── .gitignore
├── README.md
└── .venv/