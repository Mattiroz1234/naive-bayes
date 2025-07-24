from model_side.training import train_naive_bayes
from app.classifier_side.check import evaluate
from model_side.prediction import get_para_to_classify
import pandas as pd
import statistics
import os


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "data", "health_generated.csv")
    df = pd.read_csv(csv_path)
    target = "risk"
    model = None

    while True:
        print("\n--- MENU ---")
        print("1. Evaluate model accuracy")
        print("2. Predict from user input")
        print("3. Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            train = df.sample(frac=0.7)
            test = df.drop(train.index)

            model = train_naive_bayes(train, target)

            acc = evaluate(test, model, target)
            print(f"\nModel accuracy: {acc:.2%}")

        elif choice == "2":
            if model is None:
                model = train_naive_bayes(df, target)

            result = get_para_to_classify(model, target)
            print(f"\nPrediction result: {result}")

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid option. Please try again.")


def stat():
    df = pd.read_csv("data/health_generated.csv")
    target = "risk"
    a = []
    for i in range(20):
        train = df.sample(frac=0.7)
        test = df.drop(train.index)
        model = train_naive_bayes(train, target)
        acc = evaluate(test, model, target)
        a.append(f"{acc:.2%}")
    numbers = [float(p.strip('%')) for p in a]

    average = sum(numbers) / len(numbers)
    maximum = max(numbers)
    minimum = min(numbers)
    median = statistics.median(numbers)

    print(f"average: {average:.2f}%")
    print(f"maximum: {maximum:.2f}%")
    print(f"minimum: {minimum:.2f}%")
    print(f"median: {median:.2f}%")

if __name__ == "__main__":
    main()
    # for i in range(5):
    #     stat()
