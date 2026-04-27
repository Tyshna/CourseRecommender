from pydantic import BaseModel
from typing import List, Optional

class CourseEntry(BaseModel):
    course_id: str
    grade: str
    completed: bool

class StudentProfile(BaseModel):
    gpa: float
    studied_credits: Optional[int] = 0
    clicks_first_2weeks: Optional[float] = 120
    click_velocity: Optional[float] = 1.2
    zero_activity_days: Optional[float] = 5
    assessment_submission_rate: Optional[float] = 0.8
    num_of_prev_attempts: Optional[int] = 0

class EngagementProfile(BaseModel):
    score: int
    label: str



class RecommendRequest(BaseModel):
    student_profile: StudentProfile
    transcript: List[CourseEntry]
    max_credits: Optional[int] = 12
    n_recommendations: Optional[int] = 5
    method: Optional[str] = "ilp"
    subject_filter: Optional[List[str]] = []
    target_gpa: Optional[float] = None


class CourseRecommendation(BaseModel):
    course_id: str
    name: str
    subject: str
    level: int
    credits: int
    predicted_grade: float
    withdrawal_risk: float
    score: float
    prerequisite_explanation: str
    reason: str


class RecommendResponse(BaseModel):
    recommendations: List[CourseRecommendation]
    projected_final_gpa: float
    engagement_profile: Optional[EngagementProfile] = None
