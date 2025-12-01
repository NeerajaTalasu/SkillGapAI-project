from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd

from resume_reader import extract_resume_text
from jd_reader import extract_jd_text
from skill_extractor import extract_skills
from skill_compare import compute_partial_matches, calculate_match_score, generate_recommendations

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads/"
app.config['OUTPUT_FOLDER'] = "outputs/"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

dashboard_data = {}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    resume_file = request.files['resume']
    jd_file = request.files['jd']

    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    jd_path = os.path.join(app.config['UPLOAD_FOLDER'], jd_file.filename)

    resume_file.save(resume_path)
    jd_file.save(jd_path)

    resume_text = extract_resume_text(resume_path)
    jd_text = extract_jd_text(jd_path)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched, partial, missing = compute_partial_matches(resume_skills, jd_skills)
    score = calculate_match_score(jd_skills, matched, partial)
    recommendations = generate_recommendations(missing, partial, score)

    rows = []
    for skill in matched:
        rows.append({"Skill": skill, "Status": "Matched", "Similarity": 1.0})
    for pm in partial:
        rows.append({"Skill": pm["jd_skill"], "Status": "Partially Matched", "Similarity": pm["score"]})
    for skill in missing:
        rows.append({"Skill": skill, "Status": "Missing", "Similarity": 0.0})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(app.config['OUTPUT_FOLDER'], "skill_report.csv")
    df.to_csv(csv_path, index=False)

    global dashboard_data
    dashboard_data = {
        "resume": resume_skills,
        "jd": jd_skills,
        "matched_count": len(matched),
        "partial_count": len(partial),
        "missing_count": len(missing),
        "score": score,
        "partial_labels": [pm["jd_skill"] for pm in partial],
        "partial_values": [pm["score"] for pm in partial],
        "recommendations": recommendations,
        "table": rows
    }

    return redirect(url_for("dashboard"))

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html", data=dashboard_data)

@app.route("/download")
def download():
    path = os.path.join(app.config["OUTPUT_FOLDER"], "skill_report.csv")
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
