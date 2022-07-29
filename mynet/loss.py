def MSELoss(y_target, y_pred):
    sum_sq_err, count_operations = 0, 0
    for target, pred in zip(y_target, y_pred):
        for t, p in zip(target, pred):
            sum_sq_err += (t - p) ** 2
            count_operations += 1
    return sum_sq_err / count_operations

def CrossEntropyLoss(y_target, y_pred):
    loss = 0
    for target, pred in zip(y_target, y_pred):
        target_idx = target.index(max(target))
        loss += -pred[target_idx].log()
    return loss