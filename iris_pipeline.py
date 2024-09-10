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