import pandas as pd

def calc_prob(data, column, value, target, target_value):
    subset = data[data[target] == target_value]
    return len(subset[subset[column] == value]) / len(subset)


def naive_bayes_predict(data, parameters, target):
    total = len(data)
    target_op = set(data[target])

    columns = [col for col in data.columns if col not in ["id", target]]

    probability = {}

    for i in target_op:

        probability[i] = len(data[data[target] == i]) / total

        for j, x in enumerate(columns):
            p = calc_prob(data, x, parameters[j], target, i)
            if p == 0:
                p = 0.0001
            probability[i] *= p

    total_prob = sum(probability.values())
    normalized_probs = {k: v / total_prob for k, v in probability.items()}

    rounded_probs = {k: round(float(v), 3) for k, v in normalized_probs.items()}

    return max(normalized_probs, key=normalized_probs.get), rounded_probs


def naive_bayes_predict2(data, parameters, target):
    total = len(data)
    target_op = set(data[target])

    columns = [col for col in data.columns if col not in ["id", target]]

    probability = {}

    for i in target_op:

        probability[i] = len(data[data[target] == i]) / total

        for j, x in enumerate(columns):
            p = calc_prob(data, x, parameters[j], target, i)
            if p == 0:
                p = 0.0001
            probability[i] *= p

    total_prob = sum(probability.values())
    normalized_probs = {k: v / total_prob for k, v in probability.items()}
    return max(normalized_probs, key=normalized_probs.get)





