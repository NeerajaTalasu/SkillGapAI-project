from skills_list import skills

def extract_skills(text):
    text = text.lower() 
    extracted = []

    for skill in skills:
        if skill.lower() in text:
            extracted.append(skill)

    return list(set(extracted)) 
