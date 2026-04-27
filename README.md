# 🎓 Performance-Aware Course Recommender

A data-driven academic decision-support system designed to help students select an optimal course load by balancing ambition with realistic performance expectations. The system integrates machine learning, behavioral analytics, and mathematical optimization to deliver personalized and explainable recommendations.

---

## 🌟 Overview

The **Performance-Aware Course Recommender** combines predictive modeling with optimization techniques to recommend course combinations that maximize academic success while respecting real-world constraints such as credit limits and prerequisites.

The system leverages historical academic data and behavioral signals to provide insights into:
- Expected performance  
- Dropout risk  
- GPA impact  

---

## 🚀 Key Features

### 🎯 Optimization-Driven Recommendations
- Implements **Integer Linear Programming (ILP)** using :contentReference[oaicite:0]{index=0}  
- Maximizes a composite *Fit Score* subject to:
  - Credit constraints  
  - Prerequisite satisfaction  
  - Risk thresholds  

### 📈 Predictive Analytics
- **Grade Prediction**
  - Built using :contentReference[oaicite:1]{index=1}  
  - Estimates grades based on:
    - Student history  
    - Course difficulty  

- **Persistence Modeling**
  - Predicts probability of course withdrawal  
  - Uses behavioral signals such as:
    - Click velocity  
    - Assignment submission patterns  

### 📊 Student Engagement Profiling
- Categorizes students into behavioral archetypes:
  - *Ghost* → *Deeply Engaged*  
- Profiles dynamically adjust:
  - Risk scores  
  - Recommendation confidence  

### 🕸️ Prerequisite Validation Engine
- Graph-based validation system  
- Automatically:
  - Checks eligibility  
  - Explains missing prerequisites  

### 📉 GPA Projection
- Real-time cumulative GPA estimation  
- Based on predicted grades of recommended courses  

---

## 🛠️ Technology Stack

| Layer | Tools |
|------|------|
| **Frontend** | :contentReference[oaicite:2]{index=2} |
| **Backend API** | :contentReference[oaicite:3]{index=3}, :contentReference[oaicite:4]{index=4} |
| **Core Logic** | Python 3.10+, :contentReference[oaicite:5]{index=5}, :contentReference[oaicite:6]{index=6} |
| **Machine Learning** | :contentReference[oaicite:7]{index=7}, :contentReference[oaicite:8]{index=8} |
| **Optimization** | :contentReference[oaicite:9]{index=9} (COIN-OR CBC solver) |

---

## 📁 Project Structure

```text
CourseRecommender/
├── app/
│   ├── backend/          # FastAPI endpoints, schemas, and business logic
│   └── frontend/         # Streamlit UI and interaction layer
├── modules/              # Trained models, encoders, and graph utilities
├── data/                 # Raw and processed datasets
├── notebooks/            # EDA and model experimentation
├── optimizer.py          # Core optimization and scoring logic
└── requirements.txt      # Dependency manifest
````
---

## ⚙️ Setup & Installation

### Prerequisites

* Python 3.9 or higher
* pip (Python package manager)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/CourseRecommender.git
cd CourseRecommender
```

2. (Optional but recommended) Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # On macOS/Linux
.venv\Scripts\activate         # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Step 1: Start Backend API

```bash
uvicorn app.backend.main:app --reload
```

### Step 2: Launch Frontend Dashboard

```bash
streamlit run app.frontend.app.py
```
---

## 🔄 System Workflow

1. User inputs academic and engagement data via the frontend
2. Backend processes:

   * Feature engineering
   * Model inference (grade + persistence)
3. Optimization engine selects best course combination
4. Results returned with:

   * Recommended courses
   * Predicted grades
   * GPA projection
   * Risk indicators

---

## 🧠 Design Principles

* **Explainability**
  Transparent recommendations with constraint reasoning

* **Personalization**
  Tailored using both academic and behavioral signals

* **Practicality**
  Balances ambition with achievable outcomes

---

## 📌 Use Cases

* Student academic planning
* Advisor decision support systems
* Early identification of at-risk students
* GPA forecasting and performance optimization

---

## 📄 Acknowledgment

Developed as part of the **CS F320: Course Recommendation System Project**.

---
