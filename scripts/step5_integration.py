import pandas as pd
import os
from prereq_module import (
    build_graph,
    build_catalog,
    get_catalog_df,
    save_graph
)

DATA_DIR   = "data/"
OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading full courses dataset...")
courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))

difficulty_path = os.path.join(OUTPUT_DIR, "course_difficulty.csv")
if os.path.exists(difficulty_path):
    difficulty = pd.read_csv(difficulty_path)
    courses = courses.merge(difficulty[['course_id', 'avg_difficulty']],
                            on='course_id', how='left')
    print(f"Merged difficulty scores for {difficulty['course_id'].nunique()} courses.")
else:
    print("Note: difficulty scores not yet available — skipping merge.")
    courses['avg_difficulty'] = None


# ── BUILD GRAPH AND CATALOG ──────────────────────────────────
print("Building prerequisite graph...")
graph = build_graph(courses, manual_exceptions=MANUAL_EXCEPTIONS)

print("Building course catalog...")
catalog = build_catalog(courses, manual_exceptions=MANUAL_EXCEPTIONS)

# ── EXPORT FINAL CATALOG DATAFRAME ──────────────────────────
catalog_df = get_catalog_df(catalog)
catalog_path = os.path.join(OUTPUT_DIR, "course_catalog.csv")
catalog_df.to_csv(catalog_path, index=False)
print(f"\nFinal catalog saved → {catalog_path}")
print(f"  Rows: {len(catalog_df)}, Columns: {catalog_df.columns.tolist()}")
print(catalog_df.head(10).to_string())

# ── EXPORT GRAPH JSON ────────────────────────────────────────
graph_path = os.path.join(OUTPUT_DIR, "prereq_graph.json")
save_graph(graph, graph_path)

print("\n" + "="*50)
print("Member 3 outputs ready for Member 4:")
print(f"  {catalog_path}   ← course catalog dataframe")
print(f"  {graph_path}  ← prerequisite adjacency list")
print("="*50)
print("\nMember 4 can now do:")
print("  from prereq_module import load_graph, is_eligible")
print("  graph = load_graph('outputs/prereq_graph.json')")
print("  eligible = is_eligible(student_transcript, 'c4a7f0e1-...', graph)")

