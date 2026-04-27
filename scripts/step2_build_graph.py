import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the clean sample from Step 1
sample = pd.read_csv(os.path.join(OUTPUT_DIR, "sample_courses.csv"))
SAMPLE_SUBJECTS = sample['subject_code'].unique().tolist()
print(f"Building graph for: {SAMPLE_SUBJECTS}")
print(f"Total unique courses: {len(sample)}")

# Build prerequisite graph
def build_prereq_graph(df):
    graph = {}
    for subject, group in df.groupby('subject_code'):
        group_sorted = group.sort_values('level').reset_index(drop=True)
        ids = group_sorted['course_id'].tolist()
        for i, course_id in enumerate(ids):
            graph[course_id] = [] if i == 0 else [ids[i - 1]]
    return graph

prereq_graph = build_prereq_graph(sample)

print(f"\nTotal courses in graph: {len(prereq_graph)}")
print("\nSample entries:")
for cid, prereqs in list(prereq_graph.items())[:8]:
    name = sample.loc[sample['course_id'] == cid, 'name'].values[0]
    print(f"  {name[:35]:35s} → prereqs: {prereqs if prereqs else 'none'}")

# Add cross-subject exceptions
def get_course_id(df, subject, level):
    match = df[(df['subject_code'] == subject) & (df['level'] == level)]
    return match.iloc[0]['course_id'] if not match.empty else None

cs_200_id   = get_course_id(sample, 'COMP SCI', 200)
math_100_id = get_course_id(sample, 'MATH', 100)
stat_300_id = get_course_id(sample, 'STAT', 300)
math_200_id = get_course_id(sample, 'MATH', 200)

cross_exceptions = {}
if cs_200_id and math_100_id:
    cross_exceptions[cs_200_id] = [math_100_id]
    print(f"\nCross-subject: COMP SCI 200 also requires MATH 100")
if stat_300_id and math_200_id:
    cross_exceptions[stat_300_id] = [math_200_id]
    print(f"Cross-subject: STAT 300 also requires MATH 200")

for cid, extras in cross_exceptions.items():
    if cid in prereq_graph:
        prereq_graph[cid].extend(extras)

# Save graph
graph_path = os.path.join(OUTPUT_DIR, "prereq_graph.json")
with open(graph_path, 'w') as f:
    json.dump(prereq_graph, f, indent=2)
print(f"\nGraph saved → {graph_path}")

# Visualize
meta = sample.set_index('course_id')[['subject_code', 'level', 'name']].to_dict('index')
subject_x  = {'COMP SCI': 1, 'MATH': 2, 'STAT': 3}
colors     = {'COMP SCI': '#5DCAA5', 'MATH': '#AFA9EC', 'STAT': '#F0997B'}

# Only plot levels 100-400 for clarity
plot_sample = sample[sample['level'].isin([0, 100, 200, 300, 400])].copy()
plot_meta   = plot_sample.set_index('course_id')[['subject_code', 'level', 'name']].to_dict('index')

level_y = {0: 0.5, 100: 1, 200: 2, 300: 3, 400: 4}
pos = {}
counts_at = {}
for cid, info in plot_meta.items():
    key = (info['subject_code'], info['level'])
    counts_at[key] = counts_at.get(key, 0) + 1
    x = subject_x.get(info['subject_code'], 1)
    y = level_y.get(info['level'], 0)
    pos[cid] = (x, y)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 4.5)
ax.set_ylim(-0.2, 5)
ax.axis('off')
ax.set_title("Prerequisite Graph — COMP SCI, MATH, STAT\n(showing levels 100–400)", fontsize=14, fontweight='bold')

for level, y in level_y.items():
    ax.axhline(y=y, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.text(0.1, y, f"{level}-level" if level > 0 else "0-level",
            va='center', fontsize=9, color='#aaaaaa')

# Draw edges
for cid, prereqs in prereq_graph.items():
    if cid not in pos:
        continue
    x2, y2 = pos[cid]
    for pre in prereqs:
        if pre not in pos:
            continue
        x1, y1 = pos[pre]
        same = plot_meta.get(cid, {}).get('subject_code') == plot_meta.get(pre, {}).get('subject_code')
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color='#777777', lw=1.5,
                        linestyle='solid' if same else (0, (5, 3)),
                        connectionstyle='arc3,rad=0.1'
                    ), zorder=2)

# Draw nodes
for cid, (x, y) in pos.items():
    subj  = plot_meta[cid]['subject_code']
    color = colors.get(subj, '#cccccc')
    ax.scatter(x, y, s=2000, color=color, zorder=3, edgecolors='white', linewidths=2)
    ax.text(x, y, f"{subj[:4]}\n{int(plot_meta[cid]['level'])}",
            ha='center', va='center', fontsize=7, fontweight='bold', zorder=4)

for subj, x in subject_x.items():
    ax.text(x, 4.7, subj, ha='center', fontsize=12,
            fontweight='bold', color=colors[subj])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prereq_graph.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"Graph visualization saved → {OUTPUT_DIR}prereq_graph.png")
print("\nStep 2 complete! Your prereq_graph.json and prereq_graph.png are ready.")