import random


def evaluate_model(model, loss_func, X_test, y_test):
    score = 0
    for X, y in zip(X_test, y_test):
        out = model.forward([X])
        loss = loss_func([y], out)
        score += loss
    score /= len(X_test)
    return score


def train_test_split(dataset, n_X_columns, p_train=80):
    random.shuffle(dataset)
    X = [x[:n_X_columns] for x in dataset]
    y = [x[n_X_columns:] for x in dataset]
    n_train = round(p_train / 100 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, y_train, X_test, y_test


def get_random_batch(batch_size, X, y):
    if len(X) != len(y):
        raise TypeError(
                "Length of the input data (%s) has to be equal "
                "to the length of the targets (%s) "
                % (len(X), len(y))
            )
    batch_idx = [random.randint(0, len(X)-1) for i in range(batch_size)]
    return [X[idx] for idx in batch_idx], [y[idx] for idx in batch_idx]