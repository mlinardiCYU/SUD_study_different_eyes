import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

seed = 42

def split_data(data_path, train_percentage = 0.8, validation_percentage = 0.1, test_percentage = 0.1):

    if train_percentage + validation_percentage + test_percentage != 1:
        raise ValueError("Sum of percentages must be 1")
    

    df = pd.read_csv(data_path)

    X = df.text.values
    y = pd.get_dummies(df["class"])
    labels = list(y.keys())
    y = y.iloc[:,:].values

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size = 1 - train_percentage, train_size = train_percentage, random_state = seed, shuffle = True, stratify = y)

    X_validation, X_test, y_validation, y_test = train_test_split(X_rest, y_rest, test_size = test_percentage / (validation_percentage + test_percentage), train_size = validation_percentage / (validation_percentage + test_percentage), random_state = seed, shuffle = True, stratify = y_rest)

    return (X_train, X_validation, X_test, y_train, y_validation, y_test, labels)    


def split_data_test_dataset(data_path, train_percentage = 0.8, validation_percentage = 0.1, test_percentage = 0.1):

    if train_percentage + validation_percentage + test_percentage != 1:
        raise ValueError("Sum of percentages must be 1")
    

    df = pd.read_csv(data_path)

    text = df.text.values
    y = pd.get_dummies(df["class"])
    labels = list(y.keys())
    y = y.iloc[:,:].values

    dataset_names = df.dataset.values
    X = []

    for i, t in enumerate(text):
        X.append((t, dataset_names[i]))


    _, X_rest, _, y_rest = train_test_split(X, y, test_size = 1 - train_percentage, train_size = train_percentage, random_state = seed, shuffle = True, stratify = y)

    _, X_test, _, y_test = train_test_split(X_rest, y_rest, test_size = test_percentage / (validation_percentage + test_percentage), train_size = validation_percentage / (validation_percentage + test_percentage), random_state = seed, shuffle = True, stratify = y_rest)


    output = {}

    for index, sample in enumerate(X_test):

        if sample[1] in output.keys():
            output[sample[1]][0].append(sample[0])
            output[sample[1]][1].append(y_test[index])

        else:
            output[sample[1]] = ([], [])
            output[sample[1]][0].append(sample[0])
            output[sample[1]][1].append(y_test[index])


    return (output, labels)  