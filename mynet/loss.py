def MSELoss(targets, predictions):
    sum_sq_err, count_operations = 0, 0
    for target, pred in zip(targets, predictions):
        for t, p in zip(target, pred):
            sum_sq_err += (t - p) ** 2
            count_operations += 1
    return sum_sq_err / count_operations


def CrossEntropyLoss(targets, predictions, epsilon=1e-10):
    log_sum, n_operations = 0, 0
    for target, pred in zip(targets, predictions):
        target_idx = target.index(max(target))
        log_sum += (pred[target_idx] + epsilon).log()
        n_operations += 1
    return -log_sum / n_operations