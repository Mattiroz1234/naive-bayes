from .check import predict_naive_bayes

# Collects user input for features and returns predicted class
def get_para_to_classify(model, target):

    features = [col for col in model["features"] if col != target]

    user_input = {}
    for col in features:

        options = list(model["p_conditions"].values())[0][col].keys()
        print(f"{col} - options: {list(options)}")

        value = input(f"Enter value for {col}: ").strip()

        while value not in options:
            value = input("Wrong input, try again: ").strip()

        user_input[col] = value

    prediction = predict_naive_bayes(model, user_input)
    return prediction
