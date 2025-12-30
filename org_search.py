# pages/fast_search.py
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import sqlite3
import re
import json
from pathlib import Path
import google.generativeai as genai
from datetime import datetime
import csv
import io

# --- CONFIG ---
DB_PATH = os.getenv("DB_PATH", "app_data.db")  # same DB as demo.py
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))

# Configure Gemini (same env key as demo.py)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- DB helpers (lightweight, standalone) ---
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def search_reports_by_keywords(keywords, max_rows=200):
    """
    Simple SQL LIKE search across candidate_name, candidate_email, job_description,
    report_text, summary_text, recommendation. Returns list of sqlite3.Row.
    """
    conn = get_db_conn()
    cur = conn.cursor()

    # Build WHERE clause with multiple LIKE conditions for each keyword
    clauses = []
    params = []
    fields = ["candidate_name", "candidate_email", "job_description", "report_text", "summary_text", "recommendation"]
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

def fetch_all_reports(limit=500):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM interview_reports ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

# --- LLM helper (light wrapper like demo.py) ---
def get_gemini_response_for_search(prompt_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-001")
        response = model.generate_content(
            [prompt_text],
            generation_config={"temperature": 0}
        )
        return response.text
    except Exception as e:
        return f"LLM call failed: {e}"

def llm_rank_candidates(query_text, candidate_snippets, top_k=5):
    """
    Ask Gemini to read short snippets for candidates and return the top_k matching candidate names
    and short reasons. We pass candidate_snippets as a numbered list.
    """
    prompt = f"""
You are an assistant that helps recruiters find candidates.

User query (what they want): {query_text}

Below are short candidate snippets (id, name, skills/summary). Each snippet is on its own line, prefixed with an integer ID.
Use this information to return a JSON array of up to {top_k} objects with keys:
- id: the snippet ID
- name: candidate name
- score: integer 0-100 matching score
- reason: 1-2 sentence reason why this candidate matches the query

Respond ONLY with valid JSON array. Example:
[{{"id": 1, "name":"Jane Doe", "score": 92, "reason":"Has required skills A,B and 5 years experience"}} ...]
If you cannot confidently match, return an empty JSON array.
Here are the candidates:
{candidate_snippets}
"""
    resp = get_gemini_response_for_search(prompt)
    # Try to extract JSON from response
    cleaned = re.sub(r'^\s*```(?:json)?\s*', '', resp, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.IGNORECASE)
    # find first JSON array
    m = re.search(r'\[[\s\S]*\]', cleaned)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return parsed, resp
        except:
            # try direct load
            try:
                parsed = json.loads(cleaned)
                return parsed, resp
            except:
                return None, resp
    else:
        # fallback - try to parse entire cleaned as json
        try:
            parsed = json.loads(cleaned)
            return parsed, resp
        except:
            return None, resp

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Fast Search — Admin", layout="wide")
st.title("⚡ Fast Search — Find matching candidates")

# Enforce admin-only access by default
if not st.session_state.get("admin_authenticated", False):
    st.error("Admin access required. Please log in from the main app (demo.py) sidebar first.")
    st.stop()

# If we've reached here, admin is authenticated
st.markdown(f"Logged in as admin: **{st.session_state.get('admin_username', 'admin')}**")
st.markdown("Enter a description of the candidate you're looking for (skills, seniority, role). Example: `senior python backend, Django, AWS, 5+ years`.")

col1, col2 = st.columns([3, 1])
with col1:
    query_text = st.text_area("Search query (skills/requirements):", height=120)
with col2:
    use_llm = st.checkbox("Use LLM ranking (Gemini) to rank top matches", value=True)
    max_results = st.number_input("Max results to retrieve (DB filter)", min_value=5, max_value=500, value=100, step=5)
    top_k = st.number_input("Top K to show (LLM-ranked)", min_value=1, max_value=20, value=5)

# Optionally allow searching entire DB without keywords (broad)
if st.button("🔎 Search Candidates"):
    if not query_text.strip():
        st.error("Please enter a search query describing the candidate you need.")
    else:
        # build keywords (split by commas and spaces)
        raw_keywords = [k.strip() for k in re.split(r'[,\n]', query_text) if k.strip()]
        if not raw_keywords:
            raw_keywords = query_text.strip().split()
        raw_keywords = [k for k in raw_keywords if len(k) >= 2]  # ignore tiny tokens

        with st.spinner("Searching the database..."):
            rows = search_reports_by_keywords(raw_keywords, max_rows=max_results)

        st.success(f"Found {len(rows)} candidate report(s) matching keywords (basic filter).")

        if not rows:
            st.info("No matches by keyword. You can try removing filters or try broader query.")
        else:
            # show simple table of matches
            display_rows = []
            for r in rows:
                display_rows.append({
                    "id": r["id"],
                    "name": r["candidate_name"],
                    "email": r["candidate_email"],
                    "type": r["question_type"] or "N/A",
                    "created_at": r["created_at"],
                    "recommendation": r["recommendation"] if "recommendation" in r.keys() else None,
                    "confidence": r["confidence"] if "confidence" in r.keys() else None
                })
            st.dataframe(display_rows)

            # provide download CSV
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=display_rows[0].keys())
            writer.writeheader()
            writer.writerows(display_rows)
            st.download_button("⬇️ Download matches (CSV)", csv_buffer.getvalue(), file_name=f"fast_search_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

            # Show details + optional LLM ranking
            if use_llm:
                # Build short snippets for top N candidates to pass to LLM
                snippets = []
                snippet_map = {}
                max_snippet = min(len(rows), 40)  # safety limit
                for idx, r in enumerate(rows[:max_snippet], start=1):
                    s = r["summary_text"] if "summary_text" in r.keys() and r["summary_text"] else (r["report_text"] or "")
                    # trim
                    s = (s[:800] + "...") if s and len(s) > 800 else (s or "")
                    snippet = f"{idx}. ID:{r['id']} | Name:{r['candidate_name']} | SkillsSummary: {s}"
                    snippets.append(snippet)
                    snippet_map[str(idx)] = {"db_id": r["id"], "name": r["candidate_name"], "snippet": s}

                candidate_snippets_text = "\n".join(snippets)
                with st.spinner("Calling LLM to rank and pick top matches... (this may take a few seconds)"):
                    parsed_json, raw_llm = llm_rank_candidates(query_text, candidate_snippets_text, top_k=top_k)

                if parsed_json is None:
                    st.warning("LLM did not return a clean JSON array. Showing raw LLM output below for debugging.")
                    with st.expander("View raw LLM response"):
                        st.text(raw_llm)
                else:
                    st.success("LLM produced ranked results.")
                    # Map back to DB ids and display nicely
                    results_table = []
                    for obj in parsed_json:
                        # obj should have 'id' (refers to snippet index), 'name', 'score', 'reason'
                        sid = str(obj.get("id"))
                        if sid in snippet_map:
                            mapped = snippet_map[sid]
                            results_table.append({
                                "candidate_db_id": mapped["db_id"],
                                "name": obj.get("name", mapped["name"]),
                                "score": obj.get("score", 0),
                                "reason": obj.get("reason", "")
                            })
                    if results_table:
                        st.table(results_table)
                        # Expanders for matched candidates
                        for res in results_table:
                            dbid = res["candidate_db_id"]
                            with st.expander(f"{res['name']}  — Score: {res['score']}"):
                                # fetch full row by id
                                conn = get_db_conn()
                                cur = conn.cursor()
                                cur.execute("SELECT * FROM interview_reports WHERE id = ?", (dbid,))
                                row = cur.fetchone()
                                conn.close()
                                if row:
                                    st.markdown(f"**Email:** {row['candidate_email']}")
                                    if "summary_text" in row.keys() and row["summary_text"]:
                                        st.markdown("**Stored TL;DR / Summary:**")
                                        st.write(row["summary_text"])
                                    st.markdown("**Full Report (trimmed):**")
                                    rpt = (row["report_text"][:2000] + "...") if row["report_text"] and len(row["report_text"]) > 2000 else row["report_text"]
                                    st.write(rpt)
                                    if row["pdf_path"] and os.path.exists(row["pdf_path"]):
                                        with open(row["pdf_path"], "rb") as f:
                                            st.download_button(f"⬇️ Download PDF: {row['candidate_name']}", f.read(), file_name=os.path.basename(row["pdf_path"]))
                                else:
                                    st.write("Candidate details not found (id may have changed).")
                    else:
                        st.info("LLM returned an empty list (no confident matches). Consider adjusting query or removing the LLM ranking toggle.")
            else:
                # No LLM: just show expanders for the first N
                with st.expander("View matched candidate details"):
                    for r in rows[:min(len(rows), 40)]:
                        with st.expander(f"{r['candidate_name']}  ({r['candidate_email']})"):
                            if "summary_text" in r.keys() and r["summary_text"]:
                                st.markdown("**Stored TL;DR / Summary:**")
                                st.write(r["summary_text"])
                            st.markdown("**Report (trimmed):**")
                            rpt = (r["report_text"][:2000] + "...") if r["report_text"] and len(r["report_text"]) > 2000 else r["report_text"]
                            st.write(rpt)
                            if r["pdf_path"] and os.path.exists(r["pdf_path"]):
                                with open(r["pdf_path"], "rb") as f:
                                    st.download_button(f"⬇️ Download PDF: {r['candidate_name']}", f.read(), file_name=os.path.basename(r["pdf_path"]))

st.markdown("---")
st.info("Fast Search: This page reads the same database and saved summaries used by the main app. If results seem missing, check that summaries were generated in the admin dashboard first.")
