def MSELoss(y_target, y_pred):
    result = 0
    for target, pred in zip(y_target, y_pred):
        for t, p in zip(target, pred):
            result += (t - p) ** 2
    return result / len(y_target)

def CrossEntropyLoss(y_target, y_pred):
    y = y_target.index(max(y_target))
    [preds[target_idx].log() * -1 for target_idx, preds in zip(y_target, y_pred)]
    return y_pred[y].log() * -1
