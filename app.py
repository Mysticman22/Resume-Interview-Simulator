# ---------- imports & config (patched) ----------
import os
import re
import json
import random
import base64
import tempfile
from pathlib import Path
from datetime import datetime, date, timedelta

import streamlit as st               # single import
import fitz                         # PyMuPDF
import google.generativeai as genai
from gtts import gTTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# DB/auth libs
import sqlite3
import hashlib

# Ensure reports dir
DB_PATH = "app_data.db"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Safe read of Streamlit secret for Google key ---
GOOGLE_KEY = None
try:
    GOOGLE_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    # Not present in secrets; show warning but do not crash
    st.sidebar.warning("Google API key not found in st.secrets. LLM calls will not work until you add GOOGLE_API_KEY in Streamlit Secrets.")

# Configure generative AI safely (only if key exists)
if GOOGLE_KEY:
    try:
        genai.configure(api_key=GOOGLE_KEY)
    except Exception as e:
        st.sidebar.error(f"Failed to configure Google Generative AI SDK: {e}")

# ---------- helper: robust gemini call ----------
def get_gemini_response(input_text="", pdf_content="", prompt=""):
    """Call Gemini model and return a plain string (safe fallback on error)."""
    if not GOOGLE_KEY:
        return "LLM key not configured. Set GOOGLE_API_KEY in Streamlit Secrets."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content([input_text, pdf_content, prompt], generation_config={"temperature": 0})
        # Try common attributes to extract text
        if hasattr(resp, "text"):
            return str(resp.text)
        if hasattr(resp, "result"):
            return str(resp.result)
        if hasattr(resp, "candidates"):
            # join candidate texts if available
            try:
                return "\n".join([str(c.content) for c in resp.candidates])
            except Exception:
                return str(resp)
        # Fallback to stringifying the response
        return str(resp)
    except Exception as e:
        # log and return a friendly message
        st.error(f"LLM call failed: {e}")
        return f"LLM call failed: {e}"

# ============================================
#                DB & AUTH LAYER
# ============================================
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def verify_password(raw_pw: str, pw_hash: str) -> bool:
    return hash_password(raw_pw) == pw_hash

def init_db():
    conn = get_db()
    cur = conn.cursor()
    # Admin users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Interview reports table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interview_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            candidate_email TEXT,
            job_description TEXT,
            question_type TEXT,
            report_text TEXT,
            pdf_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Migration for older DBs: add question_type if it doesn't exist
    cur.execute("PRAGMA table_info(interview_reports)")
    cols = [row["name"] for row in cur.fetchall()]
    if "question_type" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN question_type TEXT")
        conn.commit()

    # --- NEW: Add columns for summarization/recommendation if not present ---
    cur.execute("PRAGMA table_info(interview_reports)")
    cols = [row["name"] for row in cur.fetchall()]
    if "summary_text" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN summary_text TEXT")
        conn.commit()
    if "recommendation" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN recommendation TEXT")
        conn.commit()
    if "confidence" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN confidence REAL")
        conn.commit()
    # --- END NEW ---

    # --- NEW: Add columns for extra candidate data ---
    # re-query columns
    cur.execute("PRAGMA table_info(interview_reports)")
    cols = [row["name"] for row in cur.fetchall()]
    if "skills" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN skills TEXT")
        conn.commit()
    if "technical_skills" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN technical_skills TEXT")
        conn.commit()
    if "experience" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN experience TEXT")
        conn.commit()
    if "expertise_types" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN expertise_types TEXT")
        conn.commit()
    if "average_score" not in cols:
        cur.execute("ALTER TABLE interview_reports ADD COLUMN average_score REAL")
        conn.commit()
    # --- END NEW EXTRA COLUMNS ---

    # Seed default admin if not exists
    cur.execute("SELECT COUNT(*) AS c FROM admin_users")
    if cur.fetchone()["c"] == 0:
        default_user = "admin"
        default_pass = "admin123"
        cur.execute(
            "INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
            (default_user, hash_password(default_pass))
        )
        conn.commit()
    conn.close()

def admin_login(username: str, password: str) -> bool:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM admin_users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return verify_password(password, row["password_hash"])

def save_report_to_db(candidate_name: str, candidate_email: str, job_desc: str, report_text: str, pdf_path: str, question_type: str = None, skills: str = None, technical_skills: str = None, experience: str = None, expertise_types: str = None, average_score: float = None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO interview_reports
        (candidate_name, candidate_email, job_description, question_type, report_text, pdf_path, created_at, skills, technical_skills, experience, expertise_types, average_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (candidate_name, candidate_email, job_desc, question_type, report_text, pdf_path, datetime.now(), skills, technical_skills, experience, expertise_types, average_score)
    )
    conn.commit()
    conn.close()

def fetch_reports(search: str = "", qtype: str = None, start_date: date = None, end_date: date = None):
    conn = get_db()
    cur = conn.cursor()
    params = []
    clauses = []
    if search:
        like = f"%{search}%"
        clauses.append("(candidate_name LIKE ? OR candidate_email LIKE ?)")
        params.extend([like, like])
    if qtype and qtype != "All":
        clauses.append("question_type = ?")
        params.append(qtype)
    if start_date:
        clauses.append("created_at >= ?")
        params.append(start_date.strftime("%Y-%m-%d") + " 00:00:00")
    if end_date:
        clauses.append("created_at <= ?")
        params.append(end_date.strftime("%Y-%m-%d") + " 23:59:59")
    where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT * FROM interview_reports {where_clause} ORDER BY created_at DESC"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows

def fetch_distinct_question_types_with_counts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT question_type, COUNT(*) as cnt
        FROM interview_reports
        WHERE question_type IS NOT NULL AND question_type != ''
        GROUP BY question_type
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [(r["question_type"], r["cnt"]) for r in rows if r["question_type"]]

def update_report_summary_in_db(report_id: int, summary_text: str, recommendation: str, confidence: float):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE interview_reports
        SET summary_text = ?, recommendation = ?, confidence = ?
        WHERE id = ?
    """, (summary_text, recommendation, confidence, report_id))
    conn.commit()
    conn.close()

init_db()
# ============================================
#             ORIGINAL APP STARTS
# ============================================
# Note: replaced dotenv usage; secrets are read from st.secrets
# configure page
st.set_page_config(page_title="Resume Screening", layout="wide")

# ensure CSS & UI block unchanged (paste your original CSS and HTML here)
custom_css = r"""
/* (keep the same CSS content you had originally) */
"""  # your large CSS from original file
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# small hero HTML (unchanged behavior)
hero_html = """
<div class="header-hero glass-card">
  <div class="brand">
    <div class="logo">RS</div>
    <div>
      <h1>Resume Screening</h1>
      <p class="small-muted">Fast. Smooth. Insightful — interview & ATS reports.</p>
    </div>
  </div>
  <div class="header-actions">
    <div class="badge">Live Interview</div>
    <div class="small-muted">Built for hiring teams</div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# ---------- NEW: SIDEBAR ADMIN NAV ----------
st.sidebar.title("Resume Screening")
st.sidebar.markdown("### How to Use:")
st.sidebar.markdown("""
1. **Enter the Job Description**  
2. **Upload multiple PDF Resumes**  
3. Click an **Analysis Button**  
4. Or try the **Live Interview Mode** 🎬  
""")

st.sidebar.markdown("---")
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    with st.sidebar.expander("👮 Admin Login"):
        admin_user = st.text_input("Username", key="adm_user")
        admin_pass = st.text_input("Password", type="password", key="adm_pass")
        if st.button("Login as Admin"):
            if admin_login(admin_user, admin_pass):
                st.session_state.admin_authenticated = True
                st.session_state.admin_username = admin_user
                st.success("Admin login successful.")
                st.experimental_rerun()
            else:
                st.error("Invalid admin credentials.")
else:
    st.sidebar.success(f"Logged in as: {st.session_state.get('admin_username', 'admin')}")
    if st.sidebar.button("Logout"):
        st.session_state.admin_authenticated = False
        st.session_state.admin_username = None
        st.experimental_rerun()

# ------------------ PAGE ROUTING ------------------ #
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["home"])[0]

# ------------------ HOME PAGE ------------------ #
if page == "home":
    # main grid: left (job desc + actions) and right (uploader)
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
        st.markdown("### 📝 Job Description")
        input_text = st.text_area("Enter Job Description:", key="input", height=200)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

        # action buttons row
        st.markdown('<div class="glass-card" style="padding:14px">', unsafe_allow_html=True)
        st.markdown("<div style='display:flex;gap:12px;flex-wrap:wrap;'>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns([1, 1, 1, 1, 1], gap="small")
        with col_btn1:
            submit1 = st.button("📄 Resume Analysis")
        with col_btn2:
            submit2 = st.button("✅ Recruitable or Not")
        with col_btn3:
            submit3 = st.button("📊 Percentage Match")
        with col_btn4:
            submit4 = st.button("🌟 Quality ")
        with col_btn5:
            submit5 = st.button("🎤 Interview")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
        st.markdown("### 📂 Upload Resumes")
        uploaded_files = st.file_uploader("Upload resumes (PDF)...", type=["pdf"], accept_multiple_files=True)
        st.markdown('<div class="small-muted" style="margin-top:8px">Tip: Drag & drop multiple resumes</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF(s) Uploaded Successfully!")

        if "resumes_text" not in st.session_state:
            st.session_state.resumes_text = {}
        for f in uploaded_files:
            if f.name not in st.session_state.resumes_text:
                try:
                    st.session_state.resumes_text[f.name] = input_pdf_setup(f)
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")

        st.session_state.job_desc = input_text

    # Live Interview nav + admin dashboard link
    left_col, right_col = st.columns([2, 1])
    with left_col:
        if st.button("🎬 Live Interview", key="live", help="Start interactive interview session"):
            if "resumes_text" in st.session_state and st.session_state.resumes_text:
                st.experimental_set_query_params(page="live")
                st.experimental_rerun()
            else:
                st.error("⚠️ Please upload at least one resume before starting Live Interview.")
    with right_col:
        if st.session_state.admin_authenticated:
            if st.button("🗂️ Open Admin Dashboard"):
                st.experimental_set_query_params(page="admin")
                st.experimental_rerun()

    # Prompts (unchanged)
    input_prompt1 = """
You are an experienced HR professional with expertise in Computer Science, Mechanical, Civil, Electronics and Telecommunication, and Electrical Engineering.
Your task is to review job descriptions and corresponding resumes, and then:

Analyze how well each resume aligns with the job description.

Highlight the strengths and weaknesses of each candidate in relation to the job requirements.
"""

    input_prompt3 = """
You are an ATS scanner with expertise . Evaluate  job description and each resume,
aligns the job description, provide a percentage match, list missing keywords,
and share your thoughts,.
"""

    input_prompt2 = """
You are an expert  ATS scanner .
Analyze  job description each resume and determine if the applicant is a 'Strong Fit' or 'Not a Fit' depending on job description  .
just tell fit or not 
,dont give extra output.if job description is not provided, remind to provide , for better result
"""

    input_prompt4 =""" You are  an expert in English and formats for resume .
your job is to analyze the clarity ,formatting, spelling mistake and grammar in the resume ,
give explaination for each clarity,formatting ,spelling mistake and grammar.
keep proper spacing between lines for each paragraph.
."""

    input_prompt5 = """You are an experienced interviwer . your job is to scan job description
     and resume and provide 20 question to the candidte for the intervirew. the questions will be based on job description and
       resume of candidate ..."""

    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = st.session_state.resumes_text.get(uploaded_file.name, "")

            if submit1:
                try:
                    response = get_gemini_response(input_prompt1, pdf_content, input_text)
                except Exception as e:
                    response = f"LLM call failed: {e}"
                st.subheader(f"📄 Analysis for {uploaded_file.name}")
                st.write(response)

            if submit3:
                try:
                    response = get_gemini_response(input_prompt3, pdf_content, input_text)
                except Exception as e:
                    response = f"LLM call failed: {e}"
                st.subheader(f"📊 Percentage Match for {uploaded_file.name}")
                st.write(response)

            if submit2:
                try:
                    response = get_gemini_response(input_prompt2, pdf_content, input_text)
                except Exception as e:
                    response = f"LLM call failed: {e}"
                st.subheader(f"✅ Recruitability for {uploaded_file.name}")
                st.write(response)

            if submit4:
                try:
                    response = get_gemini_response(input_prompt4, pdf_content, input_text)
                except Exception as e:
                    response = f"LLM call failed: {e}"
                st.subheader(f"✅ Quality for {uploaded_file.name}")
                st.write(response)

            if submit5:
                try:
                    response = get_gemini_response(input_prompt5, pdf_content, input_text)
                except Exception as e:
                    response = f"LLM call failed: {e}"
                st.subheader(f"🎤 Interview Questions for {uploaded_file.name}")
                st.write(response)

# ------------------ LIVE INTERVIEW PAGE ------------------ #
elif page == "live":
    st.markdown('<h1 style="color:#7C4DFF;">🎬 Live Interview Session</h1>', unsafe_allow_html=True)
    st.write("Answer all 10 interview questions below. Each will have audio + text. After submitting, you'll receive detailed feedback and a PDF report.")

    # Show the tip only once at the start
    if "live_tip_shown" not in st.session_state:
        st.info("💡 Tip: Select the text box and press **Windows + H** to use voice input.")
        st.session_state.live_tip_shown = True

    if st.button("⬅️ Back to Home"):
        st.experimental_set_query_params(page="home")
        st.experimental_rerun()

    # ---------- NEW: Candidate info for report ownership ----------
    st.markdown("### 👤 Candidate Details")
    candidate_name = st.text_input("Full Name", key="cand_name")
    candidate_email = st.text_input("Email", key="cand_email")

    # Define question types
    question_types = ["Personal", "Technical", "Company or Role Specific", "Case Study", "Aptitude", "HR"]

    # Initialize session state for interactive flow
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = []
    if "scores" not in st.session_state:
        st.session_state.scores = []
    if "current_q_index" not in st.session_state:
        st.session_state.current_q_index = 0
    if "interview_finished" not in st.session_state:
        st.session_state.interview_finished = False
    if "current_type" not in st.session_state:
        st.session_state.current_type = None
    # NEW: track saved reports to avoid duplicates
    if "saved_reports" not in st.session_state:
        # key: f"{candidate_email}_{question_type}" -> pdf_path
        st.session_state.saved_reports = {}

    # Select the first resume's text as interview basis
    if "resumes_text" in st.session_state and st.session_state.resumes_text:
        first_resume = list(st.session_state.resumes_text.values())[0]
    else:
        st.warning("⚠️ Please go back and upload at least one resume.")
        st.stop()

    # Helper: create TTS audio markup for a prompt
    def audio_player_for_text(text: str, q_index: int):
        temp_path = None
        try:
            tts = gTTS(text=text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_q{q_index}.mp3") as tmp_file:
                tts.save(tmp_file.name)
                temp_path = tmp_file.name
            with open(temp_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                b64 = base64.b64encode(audio_bytes).decode()
                st.markdown(
                    f"""
                    <audio controls src="data:audio/mpeg;base64,{b64}">
                        Your browser does not support the audio element.
                    </audio>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.warning(f"Audio generation failed: {e}")
        finally:
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    # Helper: generate first question
    def generate_first_question():
        q_prompt = f"""
        You are an experienced technical interviewer.
        Based on the following job description and resume, generate exactly 10 clear interview questions that are strictly of type: {st.session_state.current_type.lower()} questions.
        Ensure all questions are strictly aligned with the selected type and relevant to the job description and resume.
        - For Personal questions: Focus on the candidate's background, motivations, and soft skills (e.g., teamwork, communication).
        - For Technical questions: Focus on specific knowledge and skills relevant to the job, such as programming languages, engineering principles, or domain-specific expertise (e.g., "Write a function to find the maximum element in an array").
        - For Company or Role Specific questions: Focus on the candidate's fit for the company culture or specific role requirements.
        - For Case Study questions: Present scenarios or problems relevant to the role for the candidate to solve.
        - For Aptitude questions: Focus on problem-solving, critical thinking, logical reasoning, or numerical ability (e.g., puzzles, math problems, or logical series).
        - For HR questions: Focus on behavioral and situational questions related to workplace dynamics and ethics.
        Format as a simple numbered list (1. , 2. , 3. ...).

        Job Description: {st.session_state.get("job_desc", "")}
        Resume: {first_resume}
        """
        resp = get_gemini_response("", "", q_prompt)
        try:
            questions = [q.strip(" .") for q in resp.split("\n") if q.strip() and q.strip()[0].isdigit()]
            return questions[0] if questions else "Tell me about a project you are most proud of and why."
        except Exception:
            return "Tell me about a project you are most proud of and why."

    # Helper: generate next question
    def generate_next_question(prev_q, prev_a):
        tried = 0
        while tried < 3:
            depend_on_answer = random.choice([True, False])
            if depend_on_answer:
                prompt = f"""
                You are an experienced technical interviewer.
                Based on the job description and resume, and the candidate's last answer, ask ONE logical follow-up interview question that is strictly of type: {st.session_state.current_type.lower()} question.
                Ensure the question is strictly aligned with the selected type and relevant to the job description and resume.
                Keep it concise and focused. Do not include any numbering or extra text.

                Job Description: {st.session_state.get("job_desc", "")}
                Resume: {first_resume}

                Previous Question: {prev_q}
                Candidate's Answer: {prev_a}
                """
            else:
                prompt = f"""
                You are an experienced technical interviewer.
                Without relying on the previous answer, ask ONE new interview question that is strictly of type: {st.session_state.current_type.lower()} question that evaluates another important area for this role.
                Ensure the question is strictly aligned with the selected type and relevant to the job description and resume.
                Keep it concise and focused. Do not include any numbering or extra text.

                Job Description: {st.session_state.get("job_desc", "")}
                Resume: {first_resume}
                """
            response = get_gemini_response("", "", prompt).strip()
            line = re.sub(r"^\s*\d+[\).\s-]*", "", response).strip()
            if line and line.lower() not in [q.lower() for q in st.session_state.questions]:
                return line
            tried += 1
        return "What is a challenging problem you solved recently, and how did you approach it?"

    # Helper: evaluate answer
    def evaluate_answer(question, answer):
        eval_prompt = f"""
            Evaluate this interview answer.
            Question: {question}
            Answer: {answer}

            Give:
            1. Constructive feedback
            2. A score out of 10
            """
        feedback = get_gemini_response("", "", eval_prompt)
        score_match = re.search(r"(\d+(\.\d+)?)/10", feedback)
        score = float(score_match.group(1)) if score_match else None
        return feedback, score

    # ---------- UPDATED: build PDF (save under ./reports and return full path) ----------
    def create_pdf(filename="Interview_Report.pdf"):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        full_path = REPORTS_DIR / filename

        # if file already exists, append timestamp to avoid overwrite collisions
        if full_path.exists():
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            full_path = REPORTS_DIR / f"{full_path.stem}_{ts}{full_path.suffix}"

        doc = SimpleDocTemplate(str(full_path))
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Interview Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Type: {st.session_state.current_type}", styles['Heading1']))
        story.append(Spacer(1, 12))
        for i, (q, ans, fb, sc) in enumerate(zip(st.session_state.questions, st.session_state.answers, st.session_state.feedback, st.session_state.scores), 1):
            story.append(Paragraph(f"Q{i}: {q}", styles['Heading2']))
            story.append(Paragraph(f"Answer: {ans}", styles['Normal']))
            story.append(Paragraph(f"Feedback: {fb}", styles['Normal']))
            story.append(Paragraph(f"Score: {sc if sc else 'N/A'} / 10", styles['Normal']))
            story.append(Spacer(1, 12))
        if any(st.session_state.scores):
            existing_scores = [s for s in st.session_state.scores if s is not None]
            if existing_scores:
                avg_score = sum(existing_scores) / len(existing_scores)
                story.append(Paragraph(f"Final Score: {avg_score:.1f} / 10", styles['Heading1']))
        doc.build(story)
        return str(full_path)

    # --- NEW: Function to extract extra data using LLM ---
    def extract_extra_candidate_data(resume_text, report_text_blob):
        prompt = f"""
You are an HR assistant. Based on the resume and interview report, extract the following in JSON format:
- skills: array of general skills (e.g., ["communication", "teamwork"])
- technical_skills: array of technical skills (e.g., ["Python", "SQL"])
- experience: string describing experience based on projects (e.g., "5 years in software development with focus on web apps")
- expertise_types: array of expertise areas (e.g., ["backend development", "data analysis"])

Resume: {resume_text}
Interview Report: {report_text_blob}

Respond ONLY with the JSON object.
"""
        resp_text = get_gemini_response("", "", prompt)
        cleaned = re.sub(r'^\s*```(?:json)?\s*', '', resp_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.IGNORECASE)
        try:
            parsed = json.loads(cleaned)
            return parsed.get('skills', '[]'), parsed.get('technical_skills', '[]'), parsed.get('experience', ''), parsed.get('expertise_types', '[]')
        except:
            # fallback to empty structures
            return '[]', '[]', '', '[]'

    # If current_type is None, prompt to select type
    if st.session_state.current_type is None:
        st.markdown("### Select Interview Question Type")
        selected_type = st.selectbox("Choose the type of questions:", question_types)
        if st.button("Start Interview for Selected Type"):
            st.session_state.current_type = selected_type
            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.feedback = []
            st.session_state.scores = []
            st.session_state.current_q_index = 0
            st.session_state.interview_finished = False
            st.experimental_rerun()

        # Allow ending before starting
        if st.button("End Interview"):
            st.experimental_set_query_params(page="home")
            st.experimental_rerun()
        st.stop()

    # Ensure first question exists
    if st.session_state.current_q_index == 0 and len(st.session_state.questions) == 0:
        first_q = generate_first_question()
        st.session_state.questions.append(first_q)

    # If finished, show report and download
    if st.session_state.interview_finished:
        st.success("🎉 Interview Finished for this type! Here's your report:")
        for i, (q, ans, fb, sc) in enumerate(zip(st.session_state.questions, st.session_state.answers, st.session_state.feedback, st.session_state.scores), 1):
            st.markdown(f"<h3 style='color:#b99bff;'>Q{i}: {q}</h3>", unsafe_allow_html=True)
            st.write(f"📝 Your Answer: {ans}")
            st.write(f"💡 Feedback: {fb}")
            st.write(f"⭐ Score: {sc if sc else 'N/A'} / 10")
            st.write("---")
        if any(st.session_state.scores):
            existing_scores = [s for s in st.session_state.scores if s is not None]
            if existing_scores:
                avg_score = sum(existing_scores) / len(existing_scores)
                st.subheader(f"🏆 Final Score: {avg_score:.1f} / 10")
            else:
                avg_score = None
        else:
            avg_score = None

        # Ask user for file name
        file_name_input = st.text_input("Enter file name for the PDF report (without extension):", value=f"Interview_Report_{st.session_state.current_type.replace(' ', '_')}")

        # ---------- NEW: Avoid duplicate DB insert / duplicate downloads ----------
        if st.button("📥 Download Report as PDF"):
            if not candidate_name or not candidate_email:
                st.error("Please enter your Name and Email before downloading.")
            else:
                # Key to avoid duplicate saves for same candidate + question type in this session
                save_key = f"{candidate_email.strip().lower()}_{st.session_state.current_type.strip()}"
                requested_filename = f"{file_name_input}.pdf"

                # If already saved in this session, reuse existing path and do not save again
                if save_key in st.session_state.saved_reports:
                    existing_path = st.session_state.saved_reports[save_key]
                    if os.path.exists(existing_path):
                        st.info("Report was already generated and saved. Providing download.")
                        with open(existing_path, "rb") as f:
                            file_bytes = f.read()
                            st.download_button(f"Download '{os.path.basename(existing_path)}'", file_bytes, file_name=os.path.basename(existing_path))
                    else:
                        # file missing on disk; regenerate and update DB (safe fallback)
                        pdf_file_path = create_pdf(filename=requested_filename)

                        # Build a plain text summary to store in DB
                        lines = [f"Type: {st.session_state.current_type}\n"]
                        for i, (q, a, fb, sc) in enumerate(zip(st.session_state.questions, st.session_state.answers, st.session_state.feedback, st.session_state.scores), 1):
                            lines.append(f"Q{i}: {q}\nAnswer: {a}\nFeedback: {fb}\nScore: {sc if sc else 'N/A'}/10\n")
                        report_text_blob = "\n".join(lines)

                        # --- NEW: Extract extra data ---
                        skills, technical_skills, experience, expertise_types = extract_extra_candidate_data(first_resume, report_text_blob)

                        # SAVE TO DB (auto-upload) - now includes question_type
                        save_report_to_db(
                            candidate_name=candidate_name.strip(),
                            candidate_email=candidate_email.strip(),
                            job_desc=st.session_state.get("job_desc", ""),
                            report_text=report_text_blob,
                            pdf_path=pdf_file_path,
                            question_type=st.session_state.current_type,
                            skills=json.dumps(skills),
                            technical_skills=json.dumps(technical_skills),
                            experience=experience,
                            expertise_types=json.dumps(expertise_types),
                            average_score=avg_score
                        )
                        st.session_state.saved_reports[save_key] = pdf_file_path
                        with open(pdf_file_path, "rb") as f:
                            file_bytes = f.read()
                            st.download_button(f"Download '{os.path.basename(pdf_file_path)}'", file_bytes, file_name=os.path.basename(pdf_file_path))
                else:
                    # Not saved yet in this session: create, save, record in session_state
                    pdf_file_path = create_pdf(filename=requested_filename)

                    # Build a plain text summary to store in DB
                    lines = [f"Type: {st.session_state.current_type}\n"]
                    for i, (q, a, fb, sc) in enumerate(zip(st.session_state.questions, st.session_state.answers, st.session_state.feedback, st.session_state.scores), 1):
                        lines.append(f"Q{i}: {q}\nAnswer: {a}\nFeedback: {fb}\nScore: {sc if sc else 'N/A'}/10\n")
                    report_text_blob = "\n".join(lines)

                    # --- NEW: Extract extra data ---
                    skills, technical_skills, experience, expertise_types = extract_extra_candidate_data(first_resume, report_text_blob)

                    # SAVE TO DB (auto-upload) - now includes question_type
                    save_report_to_db(
                        candidate_name=candidate_name.strip(),
                        candidate_email=candidate_email.strip(),
                        job_desc=st.session_state.get("job_desc", ""),
                        report_text=report_text_blob,
                        pdf_path=pdf_file_path,
                        question_type=st.session_state.current_type,
                        skills=json.dumps(skills),
                        technical_skills=json.dumps(technical_skills),
                        experience=experience,
                        expertise_types=json.dumps(expertise_types),
                        average_score=avg_score
                    )

                    # record saved path so repeated clicks don't create duplicates
                    st.session_state.saved_reports[save_key] = pdf_file_path

                    # Provide download (send bytes so Streamlit doesn't re-run and re-generate inadvertently)
                    with open(pdf_file_path, "rb") as f:
                        file_bytes = f.read()
                        st.download_button(f"Download '{os.path.basename(pdf_file_path)}'", file_bytes, file_name=os.path.basename(pdf_file_path))

        # Options after download
        if st.button("Continue with Another Type"):
            st.session_state.current_type = None
            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.feedback = []
            st.session_state.scores = []
            st.session_state.current_q_index = 0
            st.session_state.interview_finished = False
            st.experimental_rerun()

        if st.button("End Interview"):
            st.experimental_set_query_params(page="home")
            st.experimental_rerun()

        st.stop()

    # Current question index and text
    i = st.session_state.current_q_index
    current_question = st.session_state.questions[i]

    # Show current question (bigger text + audio)
    st.markdown(f"<h2 style='color:#b99bff;'>❓ Question {i+1} of 10 (Type: {st.session_state.current_type})</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:30px;'>{current_question}</p>", unsafe_allow_html=True)
    audio_player_for_text(current_question, i)

    # Answer input
    ans_key = f"ans_input_{i}"
    ans = st.text_area(f"✍️ Your Answer to Q{i+1}", key=ans_key)

    # Submit current answer
    if st.button("✅ Submit Answer"):
        st.session_state.answers.append(ans)
        fb, sc = evaluate_answer(current_question, ans)
        st.session_state.feedback.append(fb)
        st.session_state.scores.append(sc)

        if len(st.session_state.questions) < 10:
            next_q = generate_next_question(current_question, ans)
            st.session_state.questions.append(next_q)

        st.session_state.current_q_index += 1

        if st.session_state.current_q_index >= 10:
            st.session_state.interview_finished = True

        st.experimental_rerun()

    # Optional: preview progress
    with st.expander("📘 Progress so far"):
        for idx in range(len(st.session_state.answers)):
            st.markdown(f"**Q{idx+1}:** {st.session_state.questions[idx]}")
            st.markdown(f"**Your Answer:** {st.session_state.answers[idx]}")
            st.markdown(f"**Feedback:** {st.session_state.feedback[idx]}")
            st.markdown(f"**Score:** {st.session_state.scores[idx] if st.session_state.scores[idx] else 'N/A'} / 10")
            st.markdown("---")

# ------------------ NEW: ADMIN DASHBOARD PAGE ------------------ #
elif page == "admin":
    st.markdown('<h1 style="color:#b99bff;">🗂️ Admin Dashboard</h1>', unsafe_allow_html=True)

    if not st.session_state.admin_authenticated:
        st.error("Admin access only. Please login from the sidebar.")
        st.stop()

    # New: allow selecting which question type to view (or All) with counts
    st.markdown("### Filter reports by type")
    types_with_counts = fetch_distinct_question_types_with_counts()
    # Build labels "Type (count)"
    available_type_labels = [f"{t} ({c})" for (t, c) in types_with_counts]
    options = ["All"] + available_type_labels

    # keep previously selected in session_state for UX
    if "admin_selected_qtype_label" not in st.session_state:
        st.session_state.admin_selected_qtype_label = "All"

    selected_label = st.selectbox(
        "Select question type to display",
        options,
        index=options.index(st.session_state.admin_selected_qtype_label) if st.session_state.admin_selected_qtype_label in options else 0
    )
    st.session_state.admin_selected_qtype_label = selected_label

    # Map label back to raw qtype (strip the " (N)" suffix). If All -> qtype=None
    if selected_label == "All":
        selected_qtype = None
    else:
        # split on " (" to be safe
        selected_qtype = selected_label.split(" (")[0].strip()

    st.markdown("---")

    # Date range filter toggle
    use_date_filter = st.checkbox("Enable date-range filter", value=False)
    start_date = None
    end_date = None
    if use_date_filter:
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Start date", value=(date.today() - timedelta(days=30)))
        with col_b:
            end_date = st.date_input("End date", value=date.today())
        if start_date and end_date and start_date > end_date:
            st.error("Start date cannot be after end date.")
            st.stop()

    # Search box
    search = st.text_input("Search by candidate name or email:", "")

    # Buttons for convenience: Clear filters
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear Filters"):
            st.session_state.admin_selected_qtype_label = "All"
            st.experimental_rerun()
    with col2:
        # Optionally show counts per type summary
        if st.button("Refresh Counts"):
            st.experimental_rerun()

    # --- NEW: improved function to generate + clean + parse LLM JSON ---
    def generate_summary_and_recommendation(report_text: str, job_desc: str):
        """
        Calls the LLM and:
          - strips markdown code fences if present
          - tries to parse JSON strictly
          - tries to extract a JSON substring if the whole response contains extra text
          - builds a human-readable summary (summary + bullets + justification) for storage/display

        Returns tuple:
          (raw_llm_text, human_summary, recommendation, confidence, parsed_json_or_None)
        """
        prompt = f"""
You are an HR assistant. Given the interview report and the job description below, produce a JSON object exactly with these keys:
- summary: one-sentence TL;DR summary.
- bullets: an array of 3 short bullet points highlighting core strengths/weaknesses.
- recommendation: one of "Hire", "Do Not Hire", or "Consider".
- justification: 1-2 sentence rationale for the recommendation.
- confidence: an integer 0-100 representing confidence percentage.

Interview Report:
{report_text}

Job Description:
{job_desc}

Respond ONLY with the JSON object (no surrounding commentary). You may or may not include markdown fences, so handle both cases.
"""
        try:
            resp_text = get_gemini_response("", report_text, prompt)
        except Exception as e:
            return (f"LLM call failed: {e}", f"LLM call failed: {e}", None, None, None)

        # 1) Remove surrounding markdown fences if any (``` or ```json
        cleaned = resp_text.strip()
        cleaned = re.sub(r'^\s*```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.IGNORECASE)

        parsed = None
        recommendation = None
        confidence_val = None

        # 2) Try strict JSON parse of cleaned text
        try:
            parsed = json.loads(cleaned)
        except Exception:
            # 3) As fallback, try to locate the first {...} JSON substring inside the raw response
            m = re.search(r'\{[\s\S]*\}', resp_text)
            if m:
                json_sub = m.group(0)
                try:
                    parsed = json.loads(json_sub)
                except Exception:
                    parsed = None

        # extract fields if parsed
        if isinstance(parsed, dict):
            recommendation = parsed.get("recommendation") or parsed.get("recommend") or None
            confidence_val = parsed.get("confidence") if parsed.get("confidence") is not None else None
            try:
                if confidence_val is not None:
                    confidence_val = float(confidence_val)
            except:
                confidence_val = None

        # Build a human-readable summary to store/display
        human_summary = ""
        if isinstance(parsed, dict):
            s = parsed.get("summary", "")
            bullets = parsed.get("bullets", []) or []
            justification = parsed.get("justification", "")
            parts = []
            if s:
                parts.append(s.strip())
            if bullets:
                parts.append("\nCore points:")
                for b in bullets:
                    parts.append(f"- {b}")
            if justification:
                parts.append("\nJustification: " + justification.strip())
            human_summary = "\n".join(parts).strip()
        else:
            # if no structured parse, store the cleaned text as fallback (without fences)
            human_summary = cleaned

        return (resp_text, human_summary, recommendation, confidence_val, parsed)
    # --- end improved function ---

    # Fetch rows with filters
    rows = fetch_reports(search=search, qtype=selected_qtype, start_date=start_date, end_date=end_date)

    if not rows:
        st.info("No reports found.")
    else:
        for r in rows:
            st.markdown("---")
            st.markdown(f"**Candidate:** {r['candidate_name']}  \n**Email:** {r['candidate_email']}  \n**Type:** {r['question_type'] or 'N/A'}  \n**Date:** {r['created_at']}")
            with st.expander("View report text"):
                st.text(r["report_text"] or "")

            pdf_path = r["pdf_path"]

            # --- Updated: three columns (Download | TL;DR & Recommend | Delete) ---
            col_download, col_summary, col_delete = st.columns([1, 1, 1])

            # Download column (unchanged)
            with col_download:
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇️ Download PDF", f.read(), file_name=os.path.basename(pdf_path), key=f"d_{r['id']}")
                else:
                    st.warning("PDF file not found on server.")

            # Summary / recommendation column (NEW UI) - NO AUTO DISPLAY of TL;DR
            with col_summary:
                existing_summary = r["summary_text"] if "summary_text" in r.keys() else None
                existing_rec = r["recommendation"] if "recommendation" in r.keys() else None
                existing_conf = r["confidence"] if "confidence" in r.keys() else None

                # BUTTON behavior:
                if existing_summary:
                    if st.button("Quick Summary", key=f"view_{r['id']}"):
                        with st.expander("TL;DR & analysis (saved)"):
                            cleaned_existing = re.sub(r'^\s*```(?:json)?\s*', '', existing_summary.strip(), flags=re.IGNORECASE)
                            cleaned_existing = re.sub(r'\s*```\s*$', '', cleaned_existing, flags=re.IGNORECASE)
                            st.markdown(cleaned_existing)
                            if existing_rec:
                                st.markdown(f"**Recommendation:** {existing_rec}  \n**Confidence:** {existing_conf if existing_conf is not None else 'N/A'}%")
                    if st.button("🔄 Regenerate Summary", key=f"regen_{r['id']}"):
                        with st.spinner("Regenerating TL;DR & recommendation..."):
                            raw_llm, human_summary, rec, conf, parsed_json = generate_summary_and_recommendation(r["report_text"] or "", r["job_description"] or "")
                            update_report_summary_in_db(r["id"], human_summary, rec or "", conf if conf is not None else None)
                            st.success("TL;DR & recommendation regenerated and saved.")
                            st.experimental_rerun()
                else:
                    if st.button("⚡ Quick Summary", key=f"gen_{r['id']}"):
                        with st.spinner("Generating TL;DR & hiring recommendation..."):
                            raw_llm, human_summary, rec, conf, parsed_json = generate_summary_and_recommendation(r["report_text"] or "", r["job_description"] or "")
                            update_report_summary_in_db(r["id"], human_summary, rec or "", conf if conf is not None else None)
                            st.success("TL;DR & recommendation generated and saved.")
                            with st.expander("Generated TL;DR & analysis"):
                                if human_summary:
                                    st.markdown(human_summary)
                                if rec:
                                    st.markdown(f"**Recommendation:** {rec}  \n**Confidence:** {conf if conf is not None else 'N/A'}%")
                            st.experimental_rerun()

            # Delete column (unchanged)
            with col_delete:
                if st.button("🗑️ Delete Report", key=f"del_{r['id']}"):
                    try:
                        # Delete DB row
                        conn = get_db()
                        cur = conn.cursor()
                        cur.execute("DELETE FROM interview_reports WHERE id = ?", (r["id"],))
                        conn.commit()
                        conn.close()
                        # Delete pdf file if exists
                        if pdf_path and os.path.exists(pdf_path):
                            try:
                                os.remove(pdf_path)
                            except Exception as e:
                                st.warning(f"Could not delete file: {e}")
                        st.success("Report deleted successfully.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to delete report: {e}")

    if st.button("⬅️ Back to Home"):
        st.experimental_set_query_params(page="home")
        st.experimental_rerun()
