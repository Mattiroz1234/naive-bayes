import pandas as pd
import random

# אפשרויות אפשריות לכל עמודה
age_groups = ["young", "middle", "senior"]
smoking = ["yes", "no"]
exercise = ["none", "light", "regular"]
diet = ["poor", "average", "good"]
alcohol = ["none", "moderate", "high"]

# פונקציית קביעת סיכון לפי מאפיינים
def classify_risk(smoke, sport, food, drink):
    risk_score = 0
    if smoke == "yes":
        risk_score += 1
    if sport == "none":
        risk_score += 1
    if food == "poor":
        risk_score += 1
    if drink == "high":
        risk_score += 1
    if risk_score >= 3:
        return "high"
    elif risk_score <= 1:
        return "low"
    else:
        return "medium"

# יצירת הנתונים
data = []
for i in range(1, 1501):
    age = random.choice(age_groups)
    smoke = random.choices(smoking, weights=[0.4, 0.6])[0]
    sport = random.choices(exercise, weights=[0.3, 0.3, 0.4])[0]
    food = random.choices(diet, weights=[0.2, 0.5, 0.3])[0]
    drink = random.choices(alcohol, weights=[0.4, 0.4, 0.2])[0]
    risk = classify_risk(smoke, sport, food, drink)

    data.append({
        "id": i,
        "age_group": age,
        "smoker": smoke,
        "exercise": sport,
        "diet": food,
        "alcohol": drink,
        "risk": risk
    })

# הפיכת רשימה ל־DataFrame
df = pd.DataFrame(data)

# שמירה לקובץ
df.to_csv("health_generated.csv", index=False)
print("✅ קובץ נוצר: health_generated.csv")
