from iris_pipeline import load_dataset, train, get_accuracy

def test_load_dataset():
    df = load_dataset()
    assert not df.empty, "The DataFrame should not be empty after loading the dataset."

def test_model_accuracy():
    df = load_dataset()
    model, X_train, X_test, y_train, y_test = train(df)
    accuracy = get_accuracy(model, X_test, y_test)
    assert accuracy > 0.8, "Model accuracy is below 80%."