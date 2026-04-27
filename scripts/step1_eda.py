import pandas as pd
import matplotlib.pyplot as plt
import re
import os

DATA_DIR   = "data/"
OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("Loading UW Madison courses table...")
print("=" * 50)

# Load CSVs
courses         = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
subjects        = pd.read_csv(os.path.join(DATA_DIR, "subjects.csv"))
subject_members = pd.read_csv(os.path.join(DATA_DIR, "subject_memberships.csv"))
offerings       = pd.read_csv(os.path.join(DATA_DIR, "course_offerings.csv"))

# Fix type mismatch: convert join keys to string
subjects['code']                = subjects['code'].astype(str)
subject_members['subject_code'] = subject_members['subject_code'].astype(str)

# Extract course level from course number
def get_level(course_number):
    match = re.search(r'\d+', str(course_number))
    if match:
        return (int(match.group()) // 100) * 100
    return None

courses['level'] = courses['number'].apply(get_level)

# Build join chain:
# offerings → subject_memberships → subjects → courses
off_subj = offerings.merge(subject_members,
                           left_on='uuid',
                           right_on='course_offering_uuid',
                           how='left')

off_subj = off_subj.merge(subjects,
                          left_on='subject_code',
                          right_on='code',
                          how='left')

merged = off_subj.merge(courses,
                        left_on='course_uuid',
                        right_on='uuid',
                        how='left')

merged = merged.rename(columns={
    'uuid_x'      : 'offering_uuid',
    'name_x'      : 'offering_name',
    'name_y'      : 'subject_name',
    'name'        : 'course_name',
    'abbreviation': 'subject_code_abbr'
})

print(f"Merged shape: {merged.shape}")

# Pick 3 sample subjects
SAMPLE_SUBJECTS = ['COMP SCI', 'MATH', 'STAT']

sample = merged[merged['subject_code_abbr'].isin(SAMPLE_SUBJECTS)].copy()
sample = sample.dropna(subset=['level'])
sample['level'] = sample['level'].astype(int)

print(f"Sample shape: {sample.shape}")

print("\nLevel distribution per subject:")
print(sample.groupby(['subject_code_abbr', 'level']).size())

# Deduplicate: keep one row per unique course
sample_unique = sample.drop_duplicates(subset=['course_uuid']).copy()
sample_unique = sample_unique[['course_uuid', 'course_name', 'number', 'level', 'subject_code_abbr']]
sample_unique = sample_unique.rename(columns={
    'course_uuid'      : 'course_id',
    'course_name'      : 'name',
    'subject_code_abbr': 'subject_code'
})

print(f"\nUnique courses shape: {sample_unique.shape}")
print(sample_unique.head(10).to_string())

# Visualize
fig, ax = plt.subplots(figsize=(8, 5))
for subj in SAMPLE_SUBJECTS:
    df_s   = sample_unique[sample_unique['subject_code'] == subj]
    counts = df_s.groupby('level').size()
    ax.plot(counts.index, counts.values, marker='o', label=subj)

ax.set_xlabel("Course Level")
ax.set_ylabel("Number of Courses")
ax.set_title("Course Count by Level — COMP SCI, MATH, STAT")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_course_counts.png"), dpi=150)
plt.show()
print(f"\nSaved: {OUTPUT_DIR}eda_course_counts.png")

# Save clean unique courses
sample_unique.to_csv(os.path.join(OUTPUT_DIR, "sample_courses.csv"), index=False)
print(f"Saved: {OUTPUT_DIR}sample_courses.csv")
print("\nStep 1 complete! Run step2_build_graph.py next.")
sample_unique.to_csv(os.path.join(OUTPUT_DIR, "sample_courses.csv"), index=False)
print(f"Saved: {OUTPUT_DIR}sample_courses.csv")