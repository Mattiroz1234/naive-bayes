import math

# Predicts the class of a single row using the Naive Bayes model
def predict_naive_bayes(model, row):
    scores = {}

    for y in model["p_base"]:
        score = math.log(model["p_base"][y])

        for feature in model["features"]:
            val = row[feature]
            prob = model["p_conditions"][y][feature].get(val, 1e-6)
            score += math.log(prob)

        scores[y] = score

    return max(scores, key=scores.get)


# Evaluates model accuracy on a DataFrame
def evaluate(df, model, target_col):
    correct = 0
    for i, row in df.iterrows():
        predi = predict_naive_bayes(model, row)
        if predi == row[target_col]:
            correct += 1
    return correct / len(df)
