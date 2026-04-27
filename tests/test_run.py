import pandas as pd

from optimizer import format_recommendation_output, recommend_courses


catalog_df = pd.DataFrame([
    {
        "course_id": "CS101",
        "subject": "Computer Sciences",
        "subject_code": "Computer Sciences",
        "level": 100,
        "credits": 3,
        "difficulty_score": 1.8
    },
    {
        "course_id": "CS201",
        "subject": "Computer Sciences",
        "subject_code": "Computer Sciences",
        "level": 200,
        "credits": 3,
        "difficulty_score": 2.1
    },
    {
        "course_id": "CS301",
        "subject": "Computer Sciences",
        "subject_code": "Computer Sciences",
        "level": 300,
        "credits": 4,
        "difficulty_score": 2.8
    },
    {
        "course_id": "MATH101",
        "subject": "Mathematics",
        "subject_code": "Mathematics",
        "level": 100,
        "credits": 3,
        "difficulty_score": 2.0
    },
    {
        "course_id": "MATH201",
        "subject": "Mathematics",
        "subject_code": "Mathematics",
        "level": 200,
        "credits": 3,
        "difficulty_score": 2.4
    },
    {
        "course_id": "STAT140",
        "subject": "Statistics",
        "subject_code": "Statistics",
        "level": 100,
        "credits": 4,
        "difficulty_score": 2.0
    },
    {
        "course_id": "STAT240",
        "subject": "Statistics",
        "subject_code": "Statistics",
        "level": 200,
        "credits": 4,
        "difficulty_score": 2.3
    }
])


test_cases = [
    {
        "name": "Case 1",
        "student_profile": {"gpa": 3.5},
        "transcript": [
            {"course_id": "CS101", "grade": "A", "completed": True},
            {"course_id": "MATH101", "grade": "B", "completed": True}
        ]
    },
    {
        "name": "Case 2",
        "student_profile": {"gpa": 2.0},
        "transcript": [
            {"course_id": "CS101", "grade": "C", "completed": True}
        ]
    },
    {
        "name": "Case 3",
        "student_profile": {"gpa": 3.8},
        "transcript": [
            {"course_id": "MATH101", "grade": "A", "completed": True}
        ]
    },
    {
        "name": "Case 4",
        "student_profile": {"gpa": 3.0},
        "transcript": []
    },
    {
        "name": "Case 5",
        "student_profile": {"gpa": 2.7},
        "transcript": [
            {"course_id": "STAT140", "grade": "B", "completed": True}
        ]
    },
    {
        "name": "Case 6",
        "student_profile": {"gpa": 3.9},
        "transcript": [
            {"course_id": "CS101", "grade": "A", "completed": True},
            {"course_id": "CS201", "grade": "A", "completed": True},
            {"course_id": "MATH101", "grade": "A", "completed": True}
        ]
    },
    {
        "name": "Case 7",
        "student_profile": {"gpa": 1.9},
        "transcript": [
            {"course_id": "MATH101", "grade": "C", "completed": True},
            {"course_id": "STAT140", "grade": "C", "completed": True}
        ]
    },
    {
        "name": "Case 8",
        "student_profile": {"gpa": 3.1},
        "transcript": [
            {"course_id": "CS101", "grade": "B", "completed": True},
            {"course_id": "STAT140", "grade": "A", "completed": True}
        ]
    },
    {
        "name": "Case 9",
        "student_profile": {"gpa": 2.4},
        "transcript": [
            {"course_id": "CS101", "grade": "B", "completed": True},
            {"course_id": "MATH101", "grade": "C", "completed": True},
            {"course_id": "STAT140", "grade": "B", "completed": True}
        ]
    },
    {
        "name": "Case 10",
        "student_profile": {"gpa": 3.6},
        "transcript": [
            {"course_id": "CS101", "grade": "A", "completed": True},
            {"course_id": "MATH101", "grade": "A", "completed": True},
            {"course_id": "STAT140", "grade": "A", "completed": True}
        ]
    }
]


for case in test_cases:
    recommendations = recommend_courses(
        case["student_profile"],
        case["transcript"],
        catalog_df,
        method="ilp",
        max_credits=12,
        n_recommendations=5
    )

    print(f"\n{case['name']}")
    print(recommendations[[
        "course_id",
        "credits",
        "predicted_grade",
        "withdrawal_risk",
        "score",
        "prerequisite_explanation",
        "reason"
    ]])
    print(format_recommendation_output(recommendations))
