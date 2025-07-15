def train_naive_bayes(df, target_col):
    model = {
        "p_base": {},
        "p_conditions": {},
        "features": [col for col in df.columns if col not in ["id", target_col]]
    }

    total = len(df)
    target_values = df[target_col].unique()

    for y in target_values:
        subset = df[df[target_col] == y]
        model["p_base"][y] = len(subset) / total
        model["p_conditions"][y] = {}

        for feature in model["features"]:
            value_counts = subset[feature].value_counts()
            total_y = len(subset)
            probs = {}

            for val in df[feature].unique():
                probs[val] = (value_counts.get(val, 0) + 1) / (total_y + len(df[feature].unique()))

            model["p_conditions"][y][feature] = probs

    return model
