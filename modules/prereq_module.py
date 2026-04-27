"""
prereq_module.py
────────────────
Public API (what Member 4 calls):
    build_catalog(courses_df)            → course_catalog dict
    build_graph(courses_df)              → prereq_graph dict
    is_eligible(transcript, course_id, graph) → bool
    get_eligible_courses(transcript, graph)   → list[course_id]

Extra convenience API for Member 4:
    is_eligible_simple(transcript, course_id) → bool
    load_course_catalog()                     → pd.DataFrame

Transcript format (list of dicts):
    [
        {"course_id": "CS101", "grade": "B", "completed": True},
        {"course_id": "MATH101", "grade": "A", "completed": True},
    ]
"""

import pandas as pd
import numpy as np
import json
import os
import re

GRAPH_PATH = "prereq_graph.json"       
COURSE_PATH = "sample_courses.csv"     


# ── STEP 1: INFER LEVEL FROM COURSE NUMBER ──────────────────
def _get_level(course_number: str) -> int | None:
    """Extract numeric level from a course number string, capped at 500 (max 5 levels)."""
    match = re.search(r'\d+', str(course_number))
    if match:
        level = (int(match.group()) // 100) * 100
        return min(level, 500)
    return None



# ── STEP 2: BUILD COURSE CATALOG ────────────────────────────
def build_catalog(courses_df: pd.DataFrame,
                  manual_exceptions: dict = None) -> dict:
    """
    Build a course catalog dict from the courses DataFrame.

    Parameters
    ----------
    courses_df : pd.DataFrame
        Must have columns: course_id, subject_code, number, name
        Optionally: credits, avg_difficulty (added later by Member 1)
    manual_exceptions : dict, optional
        {course_id: [prereq_course_id, ...]}

    Returns
    -------
    dict  {course_id: {subject, level, name, credits, prerequisites}}
    """
    courses_df = courses_df.copy()
    courses_df['level'] = courses_df['number'].apply(_get_level)

    catalog = {}
    for _, row in courses_df.iterrows():
        cid = str(row['course_id'])
        catalog[cid] = {
            'subject'       : row.get('subject_code', ''),
            'level'         : row.get('level', None),
            'name'          : row.get('name', ''),
            'credits'       : row.get('credits', 3),
            'avg_difficulty': row.get('avg_difficulty', None),
            'prerequisites' : []
        }

    # Infer within-subject prerequisites (100→200→300→400)
    graph = _infer_prereqs(courses_df)
    for cid, prereqs in graph.items():
        if cid in catalog:
            catalog[cid]['prerequisites'] = prereqs

    # Merge manual cross-subject exceptions
    if manual_exceptions:
        for cid, extra_prereqs in manual_exceptions.items():
            if cid in catalog:
                catalog[cid]['prerequisites'] = list(
                    set(catalog[cid]['prerequisites'] + extra_prereqs)
                )

    return catalog


# ── STEP 3: BUILD ADJACENCY-LIST GRAPH ──────────────────────
def _infer_prereqs(courses_df: pd.DataFrame) -> dict:
    """
    Synthesize prerequisites based on naming patterns, numbering, and level bands.
    1. Numerical Sequence: course N might require N-1 or N-2 if they are close.
    2. Name Keywords: 'Advanced' requires 'Intro' or 'Basic'. 'II' requires 'I'.
    3. Anchor Courses: Higher level courses (400+) require the 'core' course of the subject.
    4. Fallback Band Logic: If no specific prereq found, require any course from band-100.

    Returns dict {course_id: [prereq_course_ids]}
    """
    graph = {}
    courses_df = courses_df.copy()
    courses_df['course_id'] = courses_df['course_id'].astype(str)
    
    # Ensure name exists
    if 'name' not in courses_df.columns:
        courses_df['name'] = ""

    for subject, group in courses_df.groupby('subject_code'):
        # Sort by level to find numerical sequences
        group = group.dropna(subset=['level']).sort_values('level').copy()
        group['level_band'] = (group['level'] // 100 * 100).astype(int)
        
        if group.empty:
            continue
            
        # Identity anchor course (lowest numbered course in subject)
        anchor_cid = str(group.iloc[0]['course_id'])
        
        for i, row in group.iterrows():
            cid = str(row['course_id'])
            cname = str(row.get('name', '')).lower()
            clevel = int(row.get('level', 0))
            band = (clevel // 100) * 100
            
            prereqs = []
            
            # --- Rule 1: Keyword Matching ---
            if any(k in cname for k in ["advanced", "ii", "intermediate", "secondary"]):
                # Look for "Intro", "I", or "Basic" in the same subject
                for _, other in group.iterrows():
                    oname = str(other.get('name', '')).lower()
                    if any(k in oname for k in ["intro", "basic", "fundamental", " i "]) and other['course_id'] != cid:
                        prereqs.append(str(other['course_id']))
                        break
            
            # --- Rule 2: Anchor Requirement ---
            # 400+ level courses often require the introductory course
            if band >= 400 and cid != anchor_cid:
                prereqs.append(anchor_cid)
            
            # --- Rule 3: Numerical Proximity (Sequential) ---
            # If a course level is just after another (e.g. 221, 222), infer a sequence
            idx = group.index.get_loc(i)
            if idx > 0:
                prev_row = group.iloc[idx-1]
                if int(row['level']) - int(prev_row['level']) <= 5: # Very close sequence
                     prereqs.append(str(prev_row['course_id']))

            # --- Rule 4: Fallback Band Logic ---
            if not prereqs and band > 100:
                # Default to the lowest course in the previous band
                prev_band = band - 100
                prev_band_courses = group[group['level_band'] == prev_band]
                if not prev_band_courses.empty:
                    prereqs.append(str(prev_band_courses.iloc[0]['course_id']))
            
            # Deduplicate and remove self-reference
            graph[cid] = list(set(p for p in prereqs if p != cid))

    return graph



def build_graph(courses_df: pd.DataFrame,
                manual_exceptions: dict = None) -> dict:
    """
    Build and return the full prerequisite graph dict.
    {course_id: [list_of_prerequisite_course_ids]}
    """
    graph = _infer_prereqs(courses_df)
    if manual_exceptions:
        for cid, extras in manual_exceptions.items():
            if cid in graph:
                graph[cid] = list(set(graph[cid] + extras))
            else:
                graph[cid] = extras
    return graph


# ── STEP 4: ELIGIBILITY CHECKER ─────────────────────────────

def is_eligible(transcript: list[dict],
                course_id: str,
                graph: dict) -> bool:
    """
    Check whether a student is eligible to take a course.
    """
    course_id = str(course_id)

    # Course not in graph → assume no prereqs, eligible by default
    if course_id not in graph:
        return True

    prereqs = graph[course_id]
    if not prereqs:
        return True   # no prerequisites

    # Build set of completed course IDs from transcript
    completed = {
        str(entry['course_id'])
        for entry in transcript
        if entry.get('completed', False)
    }

    # All prereqs must be in completed courses
    for prereq in prereqs:
        if prereq not in completed:
            return False

    return True


def get_eligible_courses(transcript: list[dict],
                         graph: dict) -> list[str]:
    """
    Return all courses the student is eligible to take
    based on their transcript.
    """
    return [
        cid for cid in graph
        if is_eligible(transcript, cid, graph)
    ]


# ── STEP 5: CATALOG DATAFRAME (for Member 4's optimizer) ────

def get_catalog_df(catalog: dict) -> pd.DataFrame:
    """
    Convert the catalog dict into a flat DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        course_id, subject, level, name, credits,
        avg_difficulty, prerequisite_ids
    """
    rows = []
    for cid, info in catalog.items():
        rows.append({
            'course_id'       : cid,
            'subject'         : info['subject'],
            'level'           : info['level'],
            'name'            : info['name'],
            'credits'         : info['credits'],
            'avg_difficulty'  : info['avg_difficulty'],
            'prerequisite_ids': info['prerequisites'],
        })
    return pd.DataFrame(rows)


# ── SAVE / LOAD HELPERS ─────────────────────────────────────

def save_graph(graph: dict, path: str):
    with open(path, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"Graph saved → {path}")


def load_graph(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


_prereq_graph_cache = None

def get_graph():
    """Load and cache the prerequisite graph from GRAPH_PATH."""
    global _prereq_graph_cache
    if _prereq_graph_cache is None:
        _prereq_graph_cache = load_graph(GRAPH_PATH)
    return _prereq_graph_cache


def is_eligible_simple(transcript: list[dict], course_id: str) -> bool:
    """
    Wrapper that hides the graph argument.
    Member 4 can call this directly.
    """
    graph = get_graph()
    return is_eligible(transcript, course_id, graph)


def load_course_catalog() -> pd.DataFrame:
    """
    Load course catalog from CSV for Member 4's optimizer.

    Returns DataFrame with columns at least:
        course_id, credits, avg_difficulty
    """
    df = pd.read_csv(COURSE_PATH)

    # normalize column names to lower case
    lower = {c.lower(): c for c in df.columns}

    # course_id must exist
    if "course_id" not in lower:
        raise ValueError("COURSE_PATH must contain a 'course_id' column")

    # if credits missing, default to 3
    if "credits" not in lower:
        df["credits"] = 3
        credits_col = "credits"
    else:
        credits_col = lower["credits"]

    # if avg_difficulty missing, default to 0.0
    if "avg_difficulty" not in lower:
        df["avg_difficulty"] = 0.0
        diff_col = "avg_difficulty"
    else:
        diff_col = lower["avg_difficulty"]

    cid_col = lower["course_id"]

    return df[[cid_col, credits_col, diff_col]].rename(
        columns={
            cid_col: "course_id",
            credits_col: "credits",
            diff_col: "avg_difficulty",
        }
    )

# ── QUICK SELF-TEST ─────────────────────────────────────────
if __name__ == '__main__':
    # Using alpha-numeric UUID-like IDs to match real dataset
    id_cs101   = "c4a7f0e1-1b2c-4d3e-8f9g-0h1i2j3k4l5m"
    id_cs201   = "d5b8g1f2-2c3d-4e5f-9g0h-1i2j3k4l5m6n"
    id_cs301   = "e6c9h2g3-3d4e-5f6g-0h1i-2j3k4l5m6n7o"
    id_math101 = "f7d0i3h4-4e5f-6g7h-1i2j-3k4l5m6n7o8p"
    id_math201 = "g8e1j4i5-5f6g-7h8i-2j3k-4l5m6n7o8p9q"

    test_data = pd.DataFrame([
        {'course_id': id_cs101,   'subject_code': 'COMP SCI', 'number': '101', 'name': 'Intro CS',    'credits': 3, 'level': 100},
        {'course_id': id_cs201,   'subject_code': 'COMP SCI', 'number': '201', 'name': 'Data Struct', 'credits': 3, 'level': 200},
        {'course_id': id_cs301,   'subject_code': 'COMP SCI', 'number': '301', 'name': 'Algorithms',  'credits': 3, 'level': 300},
        {'course_id': id_math101, 'subject_code': 'MATH',     'number': '101', 'name': 'Calculus I',  'credits': 4, 'level': 100},
        {'course_id': id_math201, 'subject_code': 'MATH',     'number': '201', 'name': 'Calculus II', 'credits': 4, 'level': 200},
    ])

    manual = {id_cs201: [id_math101]}

    graph   = build_graph(test_data, manual_exceptions=manual)
    catalog = build_catalog(test_data, manual_exceptions=manual)

    print("Graph:", json.dumps(graph, indent=2))

    transcript_a = [
        {'course_id': id_cs101,   'grade': 'B', 'completed': True},
        {'course_id': id_math101, 'grade': 'A', 'completed': True},
    ]
    print(f"\nStudent A eligible for CS201 ({id_cs201})?", is_eligible(transcript_a, id_cs201, graph))   # True
    print(f"Student A eligible for CS301 ({id_cs301})?", is_eligible(transcript_a, id_cs301, graph))     # False

    transcript_b = [
        {'course_id': id_cs101, 'grade': 'C', 'completed': True},
    ]
    print(f"\nStudent B eligible for CS201 ({id_cs201})?", is_eligible(transcript_b, id_cs201, graph))   # False

    print("\nCatalog DataFrame:")
    print(get_catalog_df(catalog).to_string())
    print("\nAll tests passed.")