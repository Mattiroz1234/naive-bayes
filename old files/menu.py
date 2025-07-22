from help.classifies import naive_bayes_predict
import pandas as pd

def get_para_to_classify(file):
    df = pd.read_csv(file)
    print(list(df.columns))

    target = input("Select a parameter to predict: ").strip().lower()
    while target not in df.columns or target == "id":
        target = input("wrong input, try agine: ").strip().lower()

    columns = [col for col in df.columns if col != target and col != "id"]

    user_input = []
    for col in columns:
        options = df[col].unique().tolist()
        print(f"{col} - options: {options}")
        value = input(f"enter val for - {col}: ").strip().lower()

        while value not in options:
            value = input("wrong input, try agine: ").strip().lower()

        user_input.append(value)

    return naive_bayes_predict(df, user_input, target)



print(get_para_to_classify('../health_generated.csv'))