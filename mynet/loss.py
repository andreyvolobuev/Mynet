def MSELoss(y_target, y_pred):
    result = 0
    for y, y_ in zip(y_target, y_pred):
        result += (y[0] - y_[0]) ** 2
    return result / len(y_target)

def CrossEntropyLoss(y_target, y_pred):
    y = y_target.index(max(y_target))
    return y_pred[y].log() * -1