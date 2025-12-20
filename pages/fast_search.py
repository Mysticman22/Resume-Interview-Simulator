from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import sqlite3
import re
from pathlib import Path
from datetime import datetime
import csv
import io

DB_PATH = os.getenv("DB_PATH", "app_data.db")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))

def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def search_reports_by_keywords(keywords, max_rows=200):
    conn = get_db_conn()
    cur = conn.cursor()
    clauses = []
    params = []
    fields = [
        "candidate_name", "candidate_email", "job_description", "report_text",
        "summary_text", "recommendation", "skills", "technical_skills",
        "experience", "expertise_types"
    ]
    for kw in keywords:
        kw_like = f"%{kw}%"
        field_clauses = [f"{f} LIKE ?" for f in fields]
        clauses.append("(" + " OR ".join(field_clauses) + ")")
        params.extend([kw_like] * len(fields))
    where = " AND ".join(clauses) if clauses else "1"
    query = f"SELECT * FROM interview_reports WHERE {where} ORDER BY created_at DESC LIMIT ?"
    params.append(max_rows)
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows

def execute_custom_sql(sql_query):
    if not sql_query.strip().lower().startswith("select"):
        return None, "Only SELECT queries are allowed."
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute(sql_query)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

st.set_page_config(page_title="Fast Search — Admin", layout="wide")
st.title("⚡ Fast Search — Find matching candidates")

if not st.session_state.get("admin_authenticated", False):
    st.error("Admin access required. Please log in from the main app (demo.py) sidebar first.")
    st.stop()

st.markdown(f"Logged in as admin: **{st.session_state.get('admin_username', 'admin')}**")
st.markdown("Enter a description of the candidate you're looking for (skills, seniority, role). Example: `senior python backend, Django, AWS, 5+ years`.")

col1, col2 = st.columns([3, 1])
with col1:
    query_text = st.text_area("Search query (skills/requirements):", height=120)
with col2:
    max_results = st.number_input("Max results to retrieve (DB filter)", min_value=5, max_value=500, value=100, step=5)

if st.button("🔎 Search Candidates"):
    if not query_text.strip():
        st.error("Please enter a search query describing the candidate you need.")
    else:
        raw_keywords = [k.strip() for k in re.split(r"[,\n]", query_text) if k.strip()]
        if not raw_keywords:
            raw_keywords = query_text.strip().split()
        raw_keywords = [k for k in raw_keywords if len(k) >= 2]
        with st.spinner("Searching the database..."):
            rows = search_reports_by_keywords(raw_keywords, max_rows=max_results)
        st.success(f"Found {len(rows)} candidate report(s) matching keywords (basic filter).")
        if not rows:
            st.info("No matches by keyword. You can try removing filters or try broader query.")
        else:
            display_rows = []
            for r in rows:
                row = dict(r)  # ✅ convert to dict for safe .get()
                display_rows.append({
                    "id": row.get("id"),
                    "name": row.get("candidate_name"),
                    "email": row.get("candidate_email"),
                    "type": row.get("question_type") or "N/A",
                    "created_at": row.get("created_at"),
                    "recommendation": row.get("recommendation"),
                    "confidence": row.get("confidence"),
                    "skills": row.get("skills"),
                    "technical_skills": row.get("technical_skills"),
                    "experience": row.get("experience"),
                    "expertise_types": row.get("expertise_types"),
                    "average_score": row.get("average_score")
                })
            st.dataframe(display_rows)
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=display_rows[0].keys())
            writer.writeheader()
            writer.writerows(display_rows)
            st.download_button(
                "⬇️ Download matches (CSV)",
                csv_buffer.getvalue(),
                file_name=f"fast_search_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            with st.expander("View matched candidate details"):
                for r in rows[:min(len(rows), 40)]:
                    row = dict(r)  # ✅ safe conversion
                    with st.expander(f"{row.get('candidate_name')} ({row.get('candidate_email')})"):
                        if row.get("summary_text"):
                            st.markdown("**Stored TL;DR / Summary:**")
                            st.write(row.get("summary_text"))
                        st.markdown("**Report (trimmed):**")
                        rpt = (row.get("report_text")[:2000] + "...") if row.get("report_text") and len(row.get("report_text")) > 2000 else row.get("report_text")
                        st.write(rpt)
                        st.markdown("**Skills:** " + (row.get("skills") or "N/A"))
                        st.markdown("**Technical Skills:** " + (row.get("technical_skills") or "N/A"))
                        st.markdown("**Experience:** " + (row.get("experience") or "N/A"))
                        st.markdown("**Expertise Types:** " + (row.get("expertise_types") or "N/A"))
                        st.markdown("**Average Score:** " + str(row.get("average_score") or "N/A"))
                        if row.get("pdf_path") and os.path.exists(row.get("pdf_path")):
                            with open(row.get("pdf_path"), "rb") as f:
                                st.download_button(
                                    f"⬇️ Download PDF: {row.get('candidate_name')}",
                                    f.read(),
                                    file_name=os.path.basename(row.get("pdf_path"))
                                )

st.markdown("---")
st.markdown("### Custom SQL Query")
st.markdown("Enter a SELECT query to fetch specific data (e.g., SELECT * FROM interview_reports WHERE skills LIKE '%Python%').")
sql_query = st.text_area("SQL Query:", height=100)
if st.button("Run Query"):
    rows, error = execute_custom_sql(sql_query)
    if error:
        st.error(f"Error: {error}")
    else:
        if rows:
            st.success(f"Found {len(rows)} results.")
            st.dataframe([dict(row) for row in rows])
        else:
            st.info("No results found.")

st.markdown("---")
st.info("Fast Search: This page reads the same database and saved summaries used by the main app. If results seem missing, check that summaries were generated in the admin dashboard first.")
