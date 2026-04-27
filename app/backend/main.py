import sys
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from optimizer import recommend_courses
from app.backend.schemas import RecommendRequest, RecommendResponse
from modules.prereq_module import build_graph

app = FastAPI(title="Course Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load everything ONCE at startup
CATALOG_DF = pd.read_csv("outputs/course_catalog_clean.csv")

if "number" not in CATALOG_DF.columns:
    CATALOG_DF["number"] = CATALOG_DF["level"].fillna(0).astype(int).astype(str)
if "subject" not in CATALOG_DF.columns:
    CATALOG_DF["subject"] = CATALOG_DF["subject_code"]

PREREQ_GRAPH = build_graph(
    CATALOG_DF[["course_id", "subject_code", "number", "level"]].copy()
)

print(f"Catalog loaded: {len(CATALOG_DF)} courses")
print(f"Graph loaded: {len(PREREQ_GRAPH)} nodes")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    profile_dict = req.student_profile.dict()
    profile_dict["subject_filter"] = req.subject_filter
    transcript = [c.dict() for c in req.transcript]

    recs_df, projected_gpa = recommend_courses(
        student_profile=profile_dict,
        transcript=transcript,
        catalog_df=CATALOG_DF,
        prereq_graph=PREREQ_GRAPH,
        method=req.method,
        max_credits=req.max_credits,
        n_recommendations=req.n_recommendations,
        target_gpa=req.target_gpa
    )

    from optimizer import get_engagement_profile
    engagement_prof = get_engagement_profile(profile_dict)

    if recs_df.empty:
        return {
            "recommendations": [], 
            "projected_final_gpa": req.student_profile.gpa,
            "engagement_profile": engagement_prof
        }

    recommendations = []
    for _, row in recs_df.iterrows():
        recommendations.append({
            "course_id"               : str(row["course_id"]),
            "name"                    : str(row.get("name", "Unknown Course")),
            "subject"                 : str(row.get("subject", "")),
            "level"                   : int(row.get("level", 0) or 0),
            "credits"                 : int(row["credits"]),
            "predicted_grade"         : round(float(row["predicted_grade"]), 2),
            "withdrawal_risk"         : round(float(row["withdrawal_risk"]), 2),
            "score"                   : round(float(row["score"]), 4),
            "prerequisite_explanation": str(row["prerequisite_explanation"]),
            "reason"                  : str(row["reason"]),
        })

    return {
        "recommendations": recommendations, 
        "projected_final_gpa": float(projected_gpa),
        "engagement_profile": engagement_prof
    }


@app.get("/subjects")
def get_subjects():
    subjects = sorted(CATALOG_DF["subject_code"].dropna().unique().tolist())
    return {"subjects": subjects}

@app.get("/health")
def health():
    return {"status": "ok", "courses_loaded": len(CATALOG_DF)}