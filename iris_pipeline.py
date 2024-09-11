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

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_feature(df, feature):
    # Plot a histogram of one of the features
    df[feature].hist()
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

def plot_features(df):
    # Plot scatter plot of first two features.
    scatter = plt.scatter(
        df["sepal length (cm)"], df["sepal width (cm)"], c=df["species"]
    )
    plt.title("Scatter plot of the sepal features (width vs length)")
    plt.xlabel(xlabel="sepal length (cm)")
    plt.ylabel(ylabel="sepal width (cm)")
    plt.legend(
        scatter.legend_elements()[0],
        df["species_name"].unique(),
        loc="lower right",
        title="Classes",
    )
    plt.show()

def plot_model(model, X_test, y_test):
    # Plot the confusion matrix for the model
    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    iris_df = load_dataset()
    model, X_train, X_test, y_train, y_test = train(iris_df)
    accuracy = get_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    plot_feature(iris_df, "sepal length (cm)")
    plot_features(iris_df)
    plot_model(model, X_test, y_test)