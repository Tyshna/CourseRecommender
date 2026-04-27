import joblib
import json
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum

from modules.prereq_module import build_graph

grade_model     = joblib.load("modules/grade_model.joblib")
dropout_model   = joblib.load("modules/dropout_model.joblib")
subject_encoder = joblib.load("modules/subject_encoder.joblib")

with open("modules/term_map.json", "r") as f:
    term_map = json.load(f)

# ── CALIBRATED ENGAGEMENT THRESHOLDS (20/40/60/80 percentiles) ──
CLICKS_BINS   = [10, 61, 144, 302]
VELOCITY_BINS = [0.0, 0.1, 0.33, 0.77]

ENGAGEMENT_PROFILES = {
    1: "Ghost (never really logs in)",
    2: "Exam Crammer (low early activity, last-minute only)",
    3: "Occasional Follower (moderate, inconsistent)",
    4: "Regular Attendee (consistent engagement)",
    5: "Deeply Engaged (high volume + surging)",
}


# ── SCORING ──────────────────────────────────────────────────

def compute_score(predicted_grade, withdrawal_risk, difficulty_score,
                  target_gpa=None, alpha=0.5, beta=0.3, gamma=0.2):
    base_score = (
        alpha * predicted_grade
        + beta  * (1 - withdrawal_risk)
        - gamma * difficulty_score
    )
    
    if target_gpa is not None:
        # Boost courses that help reach the target GPA
        if predicted_grade >= target_gpa:
            base_score += 0.2
        elif predicted_grade < (target_gpa - 0.5):
            base_score -= 0.1 # Penalty if far below goal
            
    return base_score



# ── PREDICTION ───────────────────────────────────────────────

def _encode_subject(subject_value):
    classes = set(subject_encoder.classes_)
    if subject_value in classes:
        return int(subject_encoder.transform([subject_value])[0])
    return int(subject_encoder.transform(["Computer Sciences"])[0])


def predict_grade(student_profile, course, prereq_grades=None):
    subject_encoded = _encode_subject(
        course.get("subject_code", course.get("subject", ""))
    )
    leniency_score = course.get("leniency_score") or student_profile.get("leniency_score", 0)
    term_index     = student_profile.get("term_index", 0)
    gpa            = float(student_profile.get("gpa", 2.0))
    level          = min(int(course.get("level", 0) or 0), 500)

    X = pd.DataFrame([{
        "difficulty_score": float(course.get("difficulty_score", 0) or 0),
        "leniency_score"  : leniency_score,
        "level"           : level,
        "subject_enc"     : subject_encoded,
        "term_index"      : term_index,
    }])

    try:
        pred = float(grade_model.predict(X)[0])
    except Exception:
        pred = gpa

    # Blend with global GPA
    pred = 0.6 * pred + 0.4 * gpa

    # ── Engagement Adjustment ─────────────────────────────
    # Effort signals influence grades, especially for new subjects.
    clicks    = float(student_profile.get("clicks_first_2weeks", 120))
    sub_rate  = float(student_profile.get("assessment_submission_rate", 0.8))
    zero_days = float(student_profile.get("zero_activity_days", 5))

    # Effort Adjustment (max swing of ~0.4 grade points)
    effort_adj = (sub_rate - 0.8) * 0.8  # Scale submission rate impact
    if clicks > 300: effort_adj += 0.15
    if zero_days > 10: effort_adj -= 0.2
    
    pred += effort_adj

    # ── Specific Performance Adjustment ───────────────────
    # If the student did exceptionally well (or poorly) in the 
    # specific prerequisites for THIS course, adjust the prediction.
    if prereq_grades:
        points_map = {"A": 4.0, "AB": 3.5, "B": 3.0, "BC": 2.5, "C": 2.0, "D": 1.0, "F": 0.0}
        numeric_grades = [points_map.get(str(g).upper(), 2.0) for g in prereq_grades]
        avg_prereq_gpa = sum(numeric_grades) / len(numeric_grades)
        
        # Give 30% weight to specific prerequisite performance vs 70% to general/model prediction
        pred = 0.7 * pred + 0.3 * avg_prereq_gpa

    return float(min(max(pred, 0.0), 4.0))



def predict_withdrawal_risk(student_profile, course):
    X = pd.DataFrame([{
        "clicks_first_2weeks"       : student_profile.get("clicks_first_2weeks", 120),
        "click_velocity"            : student_profile.get("click_velocity", 1.2),
        "zero_activity_days"        : student_profile.get("zero_activity_days", 5),
        "assessment_submission_rate": student_profile.get("assessment_submission_rate", 0.8),
        "num_of_prev_attempts"      : student_profile.get("num_of_prev_attempts", 0),
        "studied_credits"           : int(student_profile.get("studied_credits", 0) or 0),
    }])


    try:
        prob = float(dropout_model.predict_proba(X)[0][1])
    except Exception:
        prob = 0.05

    # ── Heuristic Adjustment ──────────────────────────────
    # The base model only uses student engagement. We adjust 
    # slightly based on course difficulty and level so it varies.
    difficulty = float(course.get("difficulty_score", 0.5) or 0.5)
    level      = min(int(course.get("level", 100) or 100), 500)
    
    # Scale risk: influence by difficulty and level
    # This ensures the persistence score varies per course.
    variation = (difficulty * 0.04) + (level / 500 * 0.02)
    prob = prob + variation
    
    return float(min(max(prob, 0.0), 0.95))


def get_engagement_profile(student_profile: dict):
    """
    Maps raw engagement features to a 1-5 score and descriptive profile label.
    Calibrated against OULAD population percentiles.
    """
    clicks   = float(student_profile.get("clicks_first_2weeks", 120))
    velocity = float(student_profile.get("click_velocity", 1.2))

    # Scale clicks (1-5)
    c_score = 1
    for i, threshold in enumerate(CLICKS_BINS):
        if clicks >= threshold:
            c_score = i + 2
    
    # Scale velocity (1-5)
    v_score = 1
    for i, threshold in enumerate(VELOCITY_BINS):
        if velocity >= threshold:
            v_score = i + 2

    # Average and round
    final_score = int(round((c_score + v_score) / 2))
    final_score = max(1, min(5, final_score))

    return {
        "score": final_score,
        "label": ENGAGEMENT_PROFILES.get(final_score, "Unknown")
    }



# ── UTILITIES ────────────────────────────────────────────────

def generate_reason(row):
    grade = round(row['predicted_grade'], 2)
    persistence = round((1.0 - row['withdrawal_risk']) * 100)
    diff_val = float(row.get('difficulty_score', 0) or 0)
    difficulty = "High" if diff_val > 0.6 else ("Moderate" if diff_val > 0.3 else "Low")
    
    return (
        f"Expect a grade around {grade}. "
        f"Historical persistence is {persistence}%. "
        f"{difficulty} academic intensity."
    )



def get_prerequisite_status(transcript, course_id, graph, name_map=None):
    completed  = {str(e["course_id"]) for e in transcript if e.get("completed", False)}
    prereq_ids = list(graph.get(str(course_id), []))
    satisfied  = [p for p in prereq_ids if p in completed]
    missing    = [p for p in prereq_ids if p not in completed]
    eligible   = len(missing) == 0

    # Convert IDs to Names for a better explanation if name_map is provided
    if name_map:
        satisfied_display = [name_map.get(str(p), str(p)) for p in satisfied]
        missing_display   = [name_map.get(str(p), str(p)) for p in missing]
    else:
        satisfied_display = satisfied
        missing_display   = missing

    if not prereq_ids:
        explanation = "No prerequisites required"
    elif eligible:
        explanation = "Prerequisites satisfied: " + ", ".join(satisfied_display)
    else:
        explanation = "Missing prerequisites: " + ", ".join(missing_display)

    return {
        "prerequisite_ids"        : prereq_ids,
        "completed_prerequisites" : satisfied,
        "missing_prerequisites"   : missing,
        "prerequisite_explanation": explanation,
        "eligible"                : eligible,
    }



# ── CANDIDATE TABLE ──────────────────────────────────────────

def prepare_candidate_table(student_profile, transcript, catalog_df, graph, target_gpa=None):

    completed_courses = {
        str(e["course_id"]) for e in transcript if e.get("completed", False)
    }

    catalog_df = catalog_df.copy()
    if "subject_code" not in catalog_df.columns:
        catalog_df["subject_code"] = catalog_df.get("subject", "Unknown")
    if "subject" not in catalog_df.columns:
        catalog_df["subject"] = catalog_df["subject_code"]
    if "number" not in catalog_df.columns:
        catalog_df["number"] = catalog_df["level"].fillna(0).astype(int).astype(str)
    if "level" not in catalog_df.columns:
        catalog_df["level"] = None

    # Pre-compute highest completed level per subject for level-appropriateness bonus
    completed_ids = {str(e["course_id"]) for e in transcript if e.get("completed", False)}
    highest_completed_level: dict = {}
    for _, c in catalog_df.iterrows():
        if str(c["course_id"]) in completed_ids:
            subj  = str(c.get("subject_code", ""))
            lvl   = int(c.get("level", 0) or 0)
            if subj not in highest_completed_level or lvl > highest_completed_level[subj]:
                highest_completed_level[subj] = lvl

    # Identify global maturity (max level completed across all subjects)
    max_overall_level = max(highest_completed_level.values()) if highest_completed_level else 0

    # Create name lookup for prerequisite explanation
    name_map = dict(zip(catalog_df["course_id"].astype(str), catalog_df["name"].fillna("Unknown Course")))

    # Map course_id -> grade for quick lookup
    transcript_grades = {str(e["course_id"]): e["grade"] for e in transcript if "grade" in e}

    rows = []
    for _, course in catalog_df.iterrows():
        course_id         = course["course_id"]
        already_completed = str(course_id) in completed_courses
        prereq_status     = get_prerequisite_status(transcript, course_id, graph, name_map=name_map)
        
        # Extract grades for satisfied prerequisites
        satisfied_ids = prereq_status.get("completed_prerequisites", [])
        prereq_grades = [transcript_grades[pid] for pid in satisfied_ids if pid in transcript_grades]

        eligible          = prereq_status["eligible"] and not already_completed
        difficulty_score  = float(course.get("difficulty_score", 0) or 0)
        predicted_grade   = predict_grade(student_profile, course, prereq_grades=prereq_grades)
        withdrawal_risk   = predict_withdrawal_risk(student_profile, course)


        # ── Level-appropriateness bonus ───────────────────────
        subj         = str(course.get("subject_code", ""))
        course_level = int(course.get("level", 0) or 0)

        level_bonus = 0.0
        if subj in highest_completed_level:
            delta = course_level - highest_completed_level[subj]
            if delta == 100:
                level_bonus = 0.40    # Strong boost for ideal next step (e.g. 300 -> 400)
            elif delta == 0:
                level_bonus = 0.15    # Same level elective
            elif delta < 0:
                level_bonus = -0.30   # Penalty for backtracking
            else:
                level_bonus = -0.20   # Penalty for skipping levels (delta > 100)
        else:
            # New subject: if student is generally advanced, they can handle starting higher
            if max_overall_level >= 300 and course_level >= 200:
                level_bonus = 0.10

        # Global Progression Boost: If student has reached 300+, favor 300/400 courses generally
        if max_overall_level >= 300 and course_level >= 300:
            level_bonus += 0.20

        score = compute_score(predicted_grade, withdrawal_risk, difficulty_score, target_gpa=target_gpa) + level_bonus


        rows.append({
            "course_id"               : course_id,
            "name"                    : course.get("name", ""),
            "subject"                 : course.get("subject", ""),
            "subject_code"            : course.get("subject_code", ""),
            "level"                   : course.get("level", None),
            "credits"                 : int(course.get("credits", 3) or 3),
            "difficulty_score"        : difficulty_score,
            "eligible"                : eligible,
            "already_completed"       : already_completed,
            "prerequisite_ids"        : prereq_status["prerequisite_ids"],
            "completed_prerequisites" : prereq_status["completed_prerequisites"],
            "missing_prerequisites"   : prereq_status["missing_prerequisites"],
            "prerequisite_explanation": prereq_status["prerequisite_explanation"],
            "predicted_grade"         : predicted_grade,
            "withdrawal_risk"         : withdrawal_risk,
            "score"                   : score,
        })


    return pd.DataFrame(rows)


# ── OPTIMIZERS ───────────────────────────────────────────────

def ilp_optimizer(df, max_credits, n_recommendations=5):
    if df.empty:
        return df

    model = LpProblem("Course_Selection", LpMaximize)
    x     = {i: LpVariable(f"x_{i}", cat="Binary") for i in df.index}

    model += lpSum(df.loc[i, "score"]   * x[i] for i in df.index)
    model += lpSum(df.loc[i, "credits"] * x[i] for i in df.index) <= max_credits
    model += lpSum(x[i] for i in df.index) <= n_recommendations

    model.solve(PULP_CBC_CMD(msg=False))

    selected           = df[[x[i].value() == 1 for i in df.index]].copy()
    selected["reason"] = selected.apply(generate_reason, axis=1)
    return selected.reset_index(drop=True)


def greedy_optimizer(df, max_credits, n_recommendations=5):
    ranked        = df.sort_values("score", ascending=False).reset_index(drop=True)
    selected_rows = []
    total_credits = 0

    for _, row in ranked.iterrows():
        credits = int(row.get("credits", 3) or 3)
        if total_credits + credits <= max_credits:
            r           = row.to_dict()
            r["reason"] = generate_reason(r)
            selected_rows.append(r)
            total_credits += credits
        if len(selected_rows) >= n_recommendations:
            break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


# ── MAIN ENTRY POINT ─────────────────────────────────────────

def recommend_courses(student_profile, transcript, catalog_df,
                      method="ilp", max_credits=12, n_recommendations=5,
                      prereq_graph=None, target_gpa=None):


    catalog_df = catalog_df.copy()

    if "subject_code" not in catalog_df.columns and "subject" in catalog_df.columns:
        catalog_df["subject_code"] = catalog_df["subject"]
    if "number" not in catalog_df.columns:
        catalog_df["number"] = catalog_df["level"].fillna(0).astype(int).astype(str)
    if "subject" not in catalog_df.columns:
        catalog_df["subject"] = catalog_df["subject_code"]

    if prereq_graph is None:
        prereq_graph = build_graph(
            catalog_df[["course_id", "subject_code", "number", "level"]].copy()
        )

    completed_courses = {
        str(e["course_id"]) for e in transcript if e.get("completed", False)
    }

    # STEP 1 — eligibility: not already completed, prereqs met,
    #           and course level ≤ highest_completed_level + 100
    #           (students can only advance one band at a time per subject)

    # Compute highest completed level per subject
    highest_done: dict = {}
    for _, c in catalog_df.iterrows():
        if str(c["course_id"]) in completed_courses:
            subj = str(c.get("subject_code", ""))
            lvl  = int(c.get("level", 0) or 0)
            if subj not in highest_done or lvl > highest_done[subj]:
                highest_done[subj] = lvl

    def _level_allowed(cid: str) -> bool:
        """Return True if the course is within one band of the student's
        highest completed level in that subject, or the subject is new."""
        row = catalog_df[catalog_df["course_id"].astype(str) == cid]
        if row.empty:
            return True
        subj  = str(row.iloc[0].get("subject_code", ""))
        level = int(row.iloc[0].get("level", 0) or 0)
        if subj not in highest_done:
            return True   # subject not yet started → all levels open
        return level <= highest_done[subj] + 100

    eligible_ids = [
        cid for cid in catalog_df["course_id"].astype(str)
        if cid not in completed_courses
        and all(p in completed_courses for p in prereq_graph.get(str(cid), []))
        and _level_allowed(cid)
    ]

    # Include courses not in graph (no prereqs) subject to level cap
    eligible_ids = list(set(eligible_ids) | {
        cid for cid in catalog_df["course_id"].astype(str)
        if str(cid) not in prereq_graph
        and cid not in completed_courses
        and _level_allowed(cid)
    })

    eligible_catalog = catalog_df[
        catalog_df["course_id"].astype(str).isin(eligible_ids)
    ].copy()

    # STEP 2 — subject filter
    subject_filter = student_profile.get("subject_filter", [])
    if subject_filter:
        eligible_catalog = eligible_catalog[
            eligible_catalog["subject_code"].isin(subject_filter)
        ].copy()

    # STEP 3 — cap at 200 for performance (after subject filter)
    eligible_catalog = eligible_catalog.head(200)

    if eligible_catalog.empty:
        return pd.DataFrame()

    # STEP 4 — score all candidates
    candidates  = prepare_candidate_table(
        student_profile, transcript, eligible_catalog, prereq_graph, target_gpa=target_gpa
    )

    eligible_df = candidates[candidates["eligible"] == True].copy()

    if eligible_df.empty:
        return pd.DataFrame()

    # STEP 5 — optimize
    if method in {"ilp", "pulp"}:
        selected = ilp_optimizer(eligible_df, max_credits, n_recommendations)
    else:
        selected = greedy_optimizer(eligible_df, max_credits, n_recommendations)

    # STEP 6 — calculate projected cumulative GPA
    current_gpa = float(student_profile.get("gpa", 2.0))
    current_credits = int(student_profile.get("studied_credits", 0) or 0)
    
    # Weight predicted grades by credits
    recs_grade_points = sum(row["predicted_grade"] * row["credits"] for _, row in selected.iterrows())
    recs_credits = sum(row["credits"] for _, row in selected.iterrows())
    
    total_credits = current_credits + recs_credits
    if total_credits > 0:
        # Cumulative = (Total Past Points + Predicted New Points) / Total Credits
        projected_gpa = ((current_gpa * current_credits) + recs_grade_points) / total_credits
    else:
        projected_gpa = current_gpa
        
    return selected, projected_gpa



# ── OUTPUT FORMATTER ─────────────────────────────────────────

def format_recommendation_output(df):
    if df.empty:
        return []
    return df[["course_id", "score", "reason"]].to_dict(orient="records")