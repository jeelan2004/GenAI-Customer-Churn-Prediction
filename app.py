import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="GenAI Customer Churn System",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "churn_database.db"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


# -----------------------------
# Database helpers
# -----------------------------
@st.cache_data(ttl=60)
def load_customers(db_path: str) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    with sqlite3.connect(path) as conn:
        return pd.read_sql_query("SELECT * FROM customers", conn)


def get_db_schema(db_path: str) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("PRAGMA table_info(customers)").fetchall()
    return [
        {"name": row[1], "type": row[2]}
        for row in rows
    ]


def execute_read_only_sql(db_path: str, sql: str) -> pd.DataFrame:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("SQL query is empty.")

    # Only allow a single read-only SELECT/WITH query.
    if not re.match(r"^(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    blocked = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b"
    if re.search(blocked, sql, flags=re.IGNORECASE):
        raise ValueError("This query contains a database-modifying or restricted SQL command.")

    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed.")

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


# -----------------------------
# Gemini helpers
# -----------------------------
@st.cache_resource

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def generate_with_gemini(prompt: str) -> str:
    client = get_gemini_client()
    if client is None:
        return ""

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return getattr(response, "text", "") or ""
    except Exception:
        return ""


def fallback_customer_explanation(customer: dict) -> str:
    probability = float(customer.get("Churn_Probability", 0) or 0)
    risk = str(customer.get("Risk_Level", "Unknown"))
    tenure = customer.get("tenure", "N/A")
    monthly = customer.get("MonthlyCharges", "N/A")
    contract = customer.get("Contract", "N/A")

    if probability >= 0.75:
        action = "Prioritize this customer for an immediate retention campaign."
    elif probability >= 0.40:
        action = "Monitor this customer and consider a targeted retention offer."
    else:
        action = "Continue normal engagement while monitoring future risk changes."

    return f"""### Churn Risk Summary
The customer is classified as **{risk}** with an estimated churn probability of **{probability:.1%}**.

### Main Factors
- Tenure: {tenure}
- Monthly charges: {monthly}
- Contract code: {contract}

### Customer Behaviour Analysis
The model indicates a {risk.lower()} level of churn risk based on the stored customer features and model probability.

### Retention Recommendation
{action}

### Final Business Action
Use the risk score together with customer-service and contract information to decide the appropriate retention strategy."""


def customer_explanation(customer: dict) -> str:
    prompt = f"""
You are an expert telecom customer-retention analyst.
Analyze the following customer record from a machine-learning churn system.

Customer record:
{customer}

Return a concise professional report with exactly these sections:
1. Churn Risk Summary
2. Main Reasons for Churn
3. Customer Behavior Analysis
4. Retention Recommendations
5. Final Business Action

Do not invent customer information. If a numeric field is encoded, describe it as an encoded model feature rather than guessing its category.
"""
    result = generate_with_gemini(prompt)
    return result if result else fallback_customer_explanation(customer)


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";").strip()


def fallback_sql(question: str, columns: list[str]) -> str | None:
    q = question.lower().strip()
    colset = set(columns)

    if "high risk" in q and "Risk_Level" in colset:
        return "SELECT * FROM customers WHERE Risk_Level = 'High Risk'"
    if "medium risk" in q and "Risk_Level" in colset:
        return "SELECT * FROM customers WHERE Risk_Level = 'Medium Risk'"
    if "low risk" in q and "Risk_Level" in colset:
        return "SELECT * FROM customers WHERE Risk_Level = 'Low Risk'"
    if ("average monthly" in q or "avg monthly" in q) and "MonthlyCharges" in colset:
        return "SELECT AVG(MonthlyCharges) AS average_monthly_charges FROM customers"
    if ("count" in q or "how many" in q) and "Risk_Level" in colset:
        return "SELECT Risk_Level, COUNT(*) AS count FROM customers GROUP BY Risk_Level ORDER BY count DESC"
    if "highest probability" in q or "top risk" in q or "probability" in q:
        if "Churn_Probability" in colset:
            return "SELECT * FROM customers ORDER BY Churn_Probability DESC LIMIT 10"
    if "churn rate" in q and "Actual_Churn" in colset:
        return "SELECT AVG(Actual_Churn) AS churn_rate FROM customers"

    return None


def generate_sql_query(question: str, columns: list[str]) -> str:
    schema = ", ".join(columns)
    prompt = f"""
You are a SQLite analyst.

Table: customers
Columns: {schema}

Convert the user's English question into exactly ONE read-only SQLite SELECT query.
Rules:
- Return only SQL, with no markdown and no explanation.
- Use only the table customers and the listed columns.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, DETACH, or other write operations.
- If the question asks for a count, aggregate, average, maximum, minimum, grouping, or filtering, write the appropriate SELECT query.
- Do not invent columns.

User question:
{question}
"""
    result = clean_sql(generate_with_gemini(prompt))
    if result:
        return result

    fallback = fallback_sql(question, columns)
    if fallback:
        return fallback

    return "SELECT * FROM customers LIMIT 10"


# -----------------------------
# Load application data
# -----------------------------
st.title("📊 GenAI-Powered Customer Churn Prediction System")
st.caption("Machine learning risk analysis, customer insights, and natural-language SQL over the churn database.")

try:
    df = load_customers(str(DB_PATH))
except Exception as exc:
    st.error(f"Unable to load the customer database: {exc}")
    st.info("Place churn_database.db in the same folder as app.py and restart Streamlit.")
    st.stop()

if df.empty:
    st.warning("The customers table is empty.")
    st.stop()


# -----------------------------
# Dashboard
# -----------------------------
st.subheader("Customer Churn Dashboard")

high_count = int((df["Risk_Level"] == "High Risk").sum()) if "Risk_Level" in df.columns else 0
churn_rate = float(df["Actual_Churn"].mean()) if "Actual_Churn" in df.columns else 0
avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{len(df):,}")
c2.metric("High Risk Customers", f"{high_count:,}")
c3.metric("Actual Churn Rate", f"{churn_rate:.1%}")
c4.metric("Avg Monthly Charges", f"{avg_monthly:,.2f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Risk Level Distribution")
    if "Risk_Level" in df.columns:
        st.bar_chart(df["Risk_Level"].value_counts())

with right:
    st.markdown("#### Actual Churn Distribution")
    if "Actual_Churn" in df.columns:
        churn_counts = df["Actual_Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
        st.bar_chart(churn_counts)


# -----------------------------
# Data explorer
# -----------------------------
st.subheader("🔎 Customer Data Explorer")

if "Risk_Level" in df.columns:
    risk_options = ["All"] + sorted(df["Risk_Level"].dropna().unique().tolist())
    selected_risk = st.selectbox("Filter by Risk Level", risk_options)
    filtered_df = df if selected_risk == "All" else df[df["Risk_Level"] == selected_risk]
else:
    filtered_df = df

st.write(f"Showing **{len(filtered_df):,}** customer records.")
st.dataframe(filtered_df, use_container_width=True, height=400)


# -----------------------------
# AI explanation
# -----------------------------
st.subheader("🤖 AI Churn Explanation")

if "Churn_Probability" in df.columns:
    customer_options = list(range(len(df)))
    selected_index = st.selectbox(
        "Select customer row",
        customer_options,
        format_func=lambda i: f"Row {i} — Risk: {df.iloc[i].get('Risk_Level', 'Unknown')} — Probability: {float(df.iloc[i].get('Churn_Probability', 0)):.1%}",
    )

    selected_customer = df.iloc[selected_index].to_dict()

    if st.button("Generate AI Explanation", type="primary"):
        with st.spinner("Generating customer analysis..."):
            st.markdown(customer_explanation(selected_customer))
else:
    st.warning("Churn_Probability is not available in the database.")


# -----------------------------
# Text-to-SQL
# -----------------------------
st.subheader("💬 Ask Questions About Churn Data")

question = st.text_input(
    "Ask a question in plain English",
    placeholder="Example: Show high risk customers",
)

if st.button("Run Text-to-SQL Query"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        columns = df.columns.tolist()
        with st.spinner("Generating SQL query..."):
            sql_query = generate_sql_query(question, columns)

        st.markdown("**Generated SQL:**")
        st.code(sql_query, language="sql")

        try:
            result = execute_read_only_sql(str(DB_PATH), sql_query)
            st.success(f"Query returned {len(result):,} row(s).")
            st.dataframe(result, use_container_width=True)
        except Exception as exc:
            st.error(f"SQL query could not be executed: {exc}")


# -----------------------------
# Database schema
# -----------------------------
with st.expander("View Database Schema"):
    try:
        schema = get_db_schema(str(DB_PATH))
        st.dataframe(pd.DataFrame(schema), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))

st.caption("Note: AI explanations use Gemini when GEMINI_API_KEY is configured. The dashboard and SQL fallback functions continue to work without Gemini.")
