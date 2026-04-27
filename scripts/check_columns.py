import pandas as pd

# Load directly from CSV
courses  = pd.read_csv("data/courses.csv")
subjects = pd.read_csv("data/subjects.csv")

print("=== COURSES columns ===")
print(courses.columns.tolist())
print(courses.head(3))

print("\n=== SUBJECTS columns ===")
print(subjects.columns.tolist())
print(subjects.head(3))