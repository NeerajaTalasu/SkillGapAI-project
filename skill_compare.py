from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_partial_matches(resume_skills, jd_skills, threshold=0.6):
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    exact_matches = list(resume_set.intersection(jd_set))
    resume_only = list(resume_set - set(exact_matches))
    jd_only = list(jd_set - set(exact_matches))

    partial_matches = []
    used = set()

    if resume_only and jd_only:
        resume_emb = model.encode(resume_only, convert_to_tensor=True)
        jd_emb = model.encode(jd_only, convert_to_tensor=True)
        sim_matrix = util.cos_sim(jd_emb, resume_emb)

        for i, jd_skill in enumerate(jd_only):
            best_index = int(sim_matrix[i].argmax())
            score = float(sim_matrix[i][best_index])
            if score >= threshold:
                partial_matches.append({
                    "jd_skill": jd_skill,
                    "resume_skill": resume_only[best_index],
                    "score": round(score, 2)
                })
                used.add(jd_skill)

    missing = list(jd_set - set(exact_matches) - used)

    return exact_matches, partial_matches, missing


def calculate_match_score(jd_skills, exact, partial):
    if not jd_skills:
        return 0
    score = (len(exact) + len(partial) * 0.5) / len(jd_skills) * 100
    return round(score, 2)


def generate_recommendations(missing, partial, score):
    recommendations = []

    for skill in missing:
        recommendations.append(f"Consider learning '{skill}' to improve your job readiness.")

    for pm in partial:
        recommendations.append(f"Enhance your knowledge in '{pm['jd_skill']}' to strengthen your profile (related skill found: {pm['resume_skill']}).")

    if score < 50:
        recommendations.append("Overall skill alignment is low. Try focusing on core technical skills required in the job.")
    elif 50 <= score < 80:
        recommendations.append("Good profile, but improving missing and partial skills will increase your chances.")
    else:
        recommendations.append("Great match! You are highly suitable for this position.")

    return recommendations
