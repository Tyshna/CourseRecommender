import pandas as pd
import os

df = pd.read_csv("outputs/course_catalog.csv")

df = df.rename(columns={
    "course_uuid" : "course_id",
    "course_name" : "name",
    "subject_name": "subject_code",
})

# credits not in dataset — default to 3
df["credits"] = 3

# prereq_module needs a 'number' column to infer level
# reconstruct it from level (e.g. 300 → "300")
df["number"] = df["level"].fillna(0).astype(int).astype(str)

# drop the 25 rows with no course name
df = df.dropna(subset=["name"])

# save the clean version — this is what the app loads
df.to_csv("outputs/course_catalog_clean.csv", index=False)
print(f"Saved {len(df)} courses → outputs/course_catalog_clean.csv")
print(df.columns.tolist())