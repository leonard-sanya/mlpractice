import pandas as pd
from sklearn.datasets import load_iris

def load_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df["species_name"] = df.apply(
        lambda x: str(iris.target_names[int(x["species"])]), axis=1
    )
    return df

if __name__ == "__main__":
    iris_df = load_dataset()
    print(iris_df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df["species"], test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def get_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

if __name__ == "__main__":
    iris_df = load_dataset()
    model, X_train, X_test, y_train, y_test = get_accuracy(iris_df)
    accuracy = test(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")