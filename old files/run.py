from help.classifies import naive_bayes_predict2
import pandas as pd

def train_model(file, target):
    df = pd.read_csv(file)

    train_df = df.sample(frac=0.7)
    test_df = df.drop(train_df.index)

    features = [col for col in train_df.columns if col not in ["id", target]]
    correct = 0

    for idx, row in test_df.iterrows():
        params = [row[col] for col in features]
        prediction = naive_bayes_predict2(train_df, params, target)
        if prediction == row[target]:
            correct += 1

    accuracy = correct / len(test_df)
    return f"score: {accuracy:.2%}"

# print(train_model("health_generated.csv", "risk"))