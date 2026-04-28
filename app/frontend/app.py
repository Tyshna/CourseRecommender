import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Course Recommender", layout="wide")
st.title("Performance-Aware Course Recommendation System")


# ── LOAD CATALOG & SUBJECTS ───────────────────────────────────

@st.cache_data
def load_subjects():
    try:
        res = requests.get(f"{API_URL}/subjects", timeout=5)
        return res.json().get("subjects", [])
    except Exception:
        return []


@st.cache_data
def load_catalog():
    """
    Load course catalog locally for the course search dropdown.
    Falls back to an empty DataFrame if the file isn't found.
    """
    try:
        df = pd.read_csv("outputs/course_catalog_clean.csv")
        # Enforce level cap of 500
        df["level"] = df["level"].fillna(0).astype(int).clip(upper=500)
        
        df["display"] = (
            df["name"].fillna("Unnamed")
            + " — "
            + df["subject_code"].fillna("")
            + " (Level "
            + df["level"].astype(str)
            + ") ["
            + df["course_id"].astype(str).str[:4]  # Add ID suffix for uniqueness
            + "]"
        )

        return df[["course_id", "name", "subject_code", "level", "display"]].copy()
    except Exception:
        return pd.DataFrame(columns=["course_id", "name", "subject_code", "level", "display"])


all_subjects = load_subjects()
catalog_df   = load_catalog()

# Lookup dicts used by the dropdown
course_display_to_id = dict(zip(catalog_df["display"], catalog_df["course_id"].astype(str)))
course_id_to_display = dict(zip(catalog_df["course_id"].astype(str), catalog_df["display"]))
all_display_options  = sorted(catalog_df["display"].tolist())



# ── SIDEBAR: Student Profile ──────────────────────────────────

with st.sidebar:
    st.header("Student Profile")
    gpa         = st.slider("Current Cumulative GPA", 0.0, 4.0, 3.2, step=0.1)
    target_gpa  = st.slider("Target GPA for semester", 0.0, 4.0, 3.5, step=0.1)
    studied_cr  = st.number_input("Credits studied so far", 0, 300, 60)
    max_credits = st.selectbox("Max credits this semester", [9, 12, 15, 18], index=1)
    n_recs      = st.slider("Number of recommendations", 1, 8, 5)
    method      = st.radio("Optimizer method", ["ilp", "greedy"], index=0)

    # Mapping student types to calibrated behavioral data
    STUDENT_TYPES = {
        "Ghost (Very Low Engagement)": {
            "clicks": 5, "velocity": 0.05, "sub_rate": 0.2, "zero_days": 25, "desc": "Minimal login activity and missed submissions."
        },
        "Exam Crammer (Last-minute)": {
            "clicks": 35, "velocity": 0.1, "sub_rate": 0.5, "zero_days": 15, "desc": "Low early activity; tends to surge only near deadlines."
        },
        "Occasional Follower (Inconsistent)": {
            "clicks": 100, "velocity": 0.3, "sub_rate": 0.75, "zero_days": 10, "desc": "Moderate activity with some gaps in consistency."
        },
        "Regular Attendee (Consistent)": {
            "clicks": 220, "velocity": 0.6, "sub_rate": 0.9, "zero_days": 5, "desc": "Steady, reliable engagement throughout the term."
        },
        "Deeply Engaged (High Effort)": {
            "clicks": 450, "velocity": 1.5, "sub_rate": 1.0, "zero_days": 2, "desc": "Proactive participation with accelerating activity."
        }
    }

    st.subheader("Your Study Behavior")
    student_type = st.selectbox(
        "What type of student are you?",
        options=list(STUDENT_TYPES.keys()),
        index=3  # Default to Regular Attendee
    )
    
    # Extract values based on selected profile
    behavior = STUDENT_TYPES[student_type]
    clicks     = behavior["clicks"]
    velocity   = behavior["velocity"]
    sub_rate   = behavior["sub_rate"]
    zero_days  = behavior["zero_days"]
    prev_att   = 0 # Default

    st.caption(f"**Behavior Profile:** {behavior['desc']}")
    
    with st.expander("View raw behavioral data"):
        st.write(f"- Clicks: {clicks}")
        st.write(f"- Velocity: {velocity}")
        st.write(f"- Submission Rate: {sub_rate:.0%}")
        st.write(f"- Zero-activity days: {zero_days}")



    st.subheader("Subject preferences (optional)")
    selected_subjects = st.multiselect(
        "Select departments (leave empty = all)",
        options=all_subjects,
        default=[],
    )
    subject_filter = selected_subjects
    if subject_filter:
        st.caption(f"Filtering to {len(subject_filter)} dept(s): {', '.join(subject_filter)}")


# ── TRANSCRIPT STATE ──────────────────────────────────────────

# Default transcript uses real UW Madison course IDs (100-level, common depts)
DEFAULT_TRANSCRIPT = [
    {   # Algebra & Trigonometry — Mathematics, Level 100
        "course_id": "1a39869e-fbba-365e-9a29-863c3a94c000",
        "grade": "A", "completed": True,
    },
    {   # General Physics — Physics, Level 100
        "course_id": "001738d6-6195-3557-8833-83cb53fdc779",
        "grade": "B", "completed": True,
    },
    {   # Freshman Composition — ENGLISH, Level 100
        "course_id": "32361756-fb43-3c54-9aac-87263e2d7efb",
        "grade": "B", "completed": True,
    },
]

if "transcript" not in st.session_state:
    st.session_state.transcript = DEFAULT_TRANSCRIPT.copy()


# ── TRANSCRIPT EDITOR ─────────────────────────────────────────

st.subheader("Completed courses")
GRADES = ["A", "AB", "B", "BC", "C", "D", "F", "W"]

with st.expander("Edit transcript", expanded=True):

    # Column headers
    hcols = st.columns([4, 2, 1, 1])
    hcols[0].markdown("**Course** *(type to search)*")
    hcols[1].markdown("**Grade**")
    hcols[2].markdown("**Done**")

    to_remove = []

    for i, entry in enumerate(st.session_state.transcript):
        cols = st.columns([4, 2, 1, 1])

        # ── Course select dropdown ──────────────────────────
        with cols[0]:
            # Resolve current course_id → display label
            current_label = course_id_to_display.get(str(entry["course_id"]), "")
            sel_idx = all_display_options.index(current_label) if current_label in all_display_options else 0

            chosen_label = st.selectbox(
                f"Course select {i}",
                options=all_display_options,
                index=sel_idx,
                key=f"cselect_{i}",
                label_visibility="collapsed",
            )
            entry["course_id"] = course_display_to_id.get(chosen_label, entry["course_id"])

        # ── Grade ─────────────────────────────────────────────
        grade_idx      = GRADES.index(entry["grade"]) if entry["grade"] in GRADES else 2
        entry["grade"] = cols[1].selectbox(
            f"Grade {i}", GRADES,
            index=grade_idx,
            key=f"gr_{i}",
            label_visibility="collapsed",
        )

        # ── Completed checkbox ────────────────────────────────
        entry["completed"] = cols[2].checkbox(
            f"Completed {i}",
            value=entry["completed"],
            key=f"comp_{i}",
            label_visibility="collapsed",
        )

        # ── Remove button ─────────────────────────────────────
        if cols[3].button("✕", key=f"del_{i}"):
            to_remove.append(i)

    for i in reversed(to_remove):
        st.session_state.transcript.pop(i)

    # ── Add / Reset buttons ───────────────────────────────────
    btn_add, btn_reset = st.columns(2)

    if btn_add.button("＋ Add course"):
        default_id = str(catalog_df["course_id"].iloc[0]) if not catalog_df.empty else ""
        st.session_state.transcript.append(
            {"course_id": default_id, "grade": "B", "completed": True}
        )
        st.rerun()

    if btn_reset.button("↺ Reset to defaults"):
        st.session_state.transcript = DEFAULT_TRANSCRIPT.copy()
        st.rerun()


# ── RECOMMEND BUTTON ──────────────────────────────────────────

if st.button("Get Recommendations", type="primary", use_container_width=True):
    payload = {
        "student_profile": {
            "gpa"                       : gpa,
            "studied_credits"           : int(studied_cr),
            "clicks_first_2weeks"       : clicks,
            "click_velocity"            : velocity,
            "zero_activity_days"        : zero_days,
            "assessment_submission_rate": sub_rate,
            "num_of_prev_attempts"      : int(prev_att),
        },


        "transcript"       : [e for e in st.session_state.transcript if e["course_id"]],
        "max_credits"      : max_credits,
        "n_recommendations": n_recs,
        "method"           : method,
        "subject_filter"   : subject_filter,
        "target_gpa"       : target_gpa,
    }


    with st.spinner("Running optimizer..."):
        try:
            res = requests.post(f"{API_URL}/recommend", json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            recs = data["recommendations"]
            projected_gpa = data["projected_final_gpa"]
            projected_sem_gpa = data.get("projected_semester_gpa", 0.0)
            eng_prof = data.get("engagement_profile", {"score": 3, "label": "Moderate"})


            if not recs:
                st.warning(
                    "No courses found matching your filters. "
                    "Try broadening your subject filter or increasing max credits."
                )
            else:
                st.subheader(f"Top {len(recs)} recommended courses")

                for r in recs:
                    grade_color = "🟢" if r["predicted_grade"] >= 3.0 else ("🟡" if r["predicted_grade"] >= 2.0 else "🔴")
                    
                    # Reframe Withdrawal Risk as Persistence Signal (Inverted for positive framing)
                    persistence_val = 1.0 - r["withdrawal_risk"]
                    # 90%+ Persistence = Green, 70-90% = Amber, <70% = Blue (Challenging)
                    pers_color = "🟢" if persistence_val >= 0.85 else ("🟡" if persistence_val >= 0.70 else "🔵")

                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                        c1.markdown(f"**{r.get('name', 'Unknown Course')}**")
                        c1.caption(f"ID: {r['course_id']}")
                        c1.caption(f"{r['subject']} · Level {r['level']} · {r['credits']} credits")

                        c2.metric("Predicted Grade", f"{grade_color} {r['predicted_grade']:.2f}")
                        c3.metric(
                            "Student Persistence", 
                            f"{pers_color} {persistence_val:.0%}",
                            help="The percentage of students who successfully completed this course."
                        )
                        c4.metric(
                            "Withdrawal Risk", 
                            f"{r['withdrawal_risk']:.0%}",
                            delta=f"{r['withdrawal_risk']:.0%}",
                            delta_color="inverse",
                            help="The predicted probability of dropping out based on engagement signals."
                        )

                        st.caption(f"Fit Score: {r['score']:.3f} | 📋 {r['prerequisite_explanation']}")
                        st.caption(f"💡 {r['reason']}")

                
                # Show Projected Cumulative GPA
                st.divider()
                st.subheader("Projected Semester Result")
                p_color = "normal" if projected_gpa >= gpa else "inverse"
                gpa_delta = projected_gpa - gpa
                
                c1, c2 = st.columns([1, 1])
                c1.metric(
                    label="Projected Cumulative GPA",
                    value=f"{projected_gpa:.2f}",
                    delta=f"{gpa_delta:+.2f}",
                    delta_color=p_color,
                    help="The estimated total cumulative GPA you will have after completing the above recommended courses."
                )
                if projected_gpa >= target_gpa:
                    c2.success(f"On track to reach your target of {target_gpa:.2f}!")
                else:
                    c2.info(f"Aiming for {target_gpa:.2f}. These recommendations maximize your chances.")

                # ── STUDY HABIT SUGGESTION ──────────────────────
                midpoint = (gpa + target_gpa) / 2
                if projected_sem_gpa < midpoint:
                    st.markdown("---")
                    with st.container(border=True):
                        st.markdown("### 💡 Academic Performance Tip")
                        st.write(
                            "Based on your current engagement levels and course selection, "
                            "your predicted semester performance is lower than the midpoint "
                            "of your current and target GPA."
                        )
                        st.info(
                            "**Suggestion:** Try changing your study habits or engagement behavior "
                            "(e.g., increasing your click velocity or submission rate) to score better "
                            "and bridge the gap towards your target!"
                        )


        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")